#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, NX, NY, TMAX, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (if any)
#include "fdtd-2d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize input arrays, translated from the original C init_array().
 *
 * C prototype:
 *   void init_array (int tmax, int nx, int ny,
 *                    DATA_TYPE ex[nx][ny],
 *                    DATA_TYPE ey[nx][ny],
 *                    DATA_TYPE hz[nx][ny],
 *                    DATA_TYPE _fict_[tmax])
 *
 * In Halide, we represent:
 *   ex,ey,hz as Buffer<num_t, 2> with shape (ny, nx) and access ex(j, i)
 *   _fict_   as Buffer<num_t, 1> with shape (tmax)
 */
static void init_array(int tmax,
                       int nx,
                       int ny,
                       Halide::Buffer<num_t, 2> ex,
                       Halide::Buffer<num_t, 2> ey,
                       Halide::Buffer<num_t, 2> hz,
                       Halide::Buffer<num_t, 1> fict) {
    Var i("i"), j("j"), t("t");

    Func init_ex("init_ex"), init_ey("init_ey"), init_hz("init_hz"), init_fict("init_fict");

    // _fict_[i] = (DATA_TYPE) i;
    init_fict(t) = cast<num_t>(t);

    // ex[i][j] = (i * (j+1)) / nx;
    // ey[i][j] = (i * (j+2)) / ny;
    // hz[i][j] = (i * (j+3)) / nx;
    //
    // Remember: Buffer is indexed as (j, i) == C's [i][j].
    init_ex(j, i) = cast<num_t>(i * (j + 1)) / cast<int>(nx);
    init_ey(j, i) = cast<num_t>(i * (j + 2)) / cast<int>(ny);
    init_hz(j, i) = cast<num_t>(i * (j + 3)) / cast<int>(nx);

    // Simple but effective schedule for initialization:
    // parallelize over rows (i) and vectorize along columns (j).
    Target init_target = get_jit_target_from_environment();
    const int vec_width = init_target.natural_vector_size(type_of<num_t>());

    // 1D init for fict: just vectorize.
    init_fict.compute_root()
             .vectorize(t, vec_width);

    init_ex.compute_root()
           .parallel(i)
           .vectorize(j, vec_width);

    init_ey.compute_root()
           .parallel(i)
           .vectorize(j, vec_width);

    init_hz.compute_root()
           .parallel(i)
           .vectorize(j, vec_width);

    // Realize into the provided Buffers.
    init_fict.realize(fict);
    init_ex.realize(ex);
    init_ey.realize(ey);
    init_hz.realize(hz);
}

int main(int argc, char *argv[]) {
    // Problem sizes from PolyBench
    int tmax = TMAX;
    int nx   = NX;
    int ny   = NY;

    // Buffers for ex, ey, hz.
    //
    // C layout: ex[nx][ny] with ex[i][j]
    // Halide layout: (ny, nx) so that ex(j, i) corresponds to ex[i][j].
    Halide::Buffer<num_t, 2> exA(ny, nx);
    Halide::Buffer<num_t, 2> eyA(ny, nx);
    Halide::Buffer<num_t, 2> hzA(ny, nx);

    // Second set of buffers for double-buffering in time.
    Halide::Buffer<num_t, 2> exB(ny, nx);
    Halide::Buffer<num_t, 2> eyB(ny, nx);
    Halide::Buffer<num_t, 2> hzB(ny, nx);

    // _fict_[t] array
    Halide::Buffer<num_t, 1> fict(tmax);

    // Initialize arrays
    init_array(tmax, nx, ny, exA, eyA, hzA, fict);

    // Build the Halide pipeline that performs a single time-step.
    //
    // Inputs for the pipeline (current time-step fields)
    ImageParam ex_param(type_of<num_t>(), 2, "ex_param");
    ImageParam ey_param(type_of<num_t>(), 2, "ey_param");
    ImageParam hz_param(type_of<num_t>(), 2, "hz_param");

    // Scalar parameter for _fict_[t] at the current time-step.
    Param<num_t> fict_param("fict_param");

    // Vars: j is the fast (x/column), i is the slow (y/row).
    Var j("j"), i("i");

    // Constants SCALAR_VAL(0.5) and SCALAR_VAL(0.7), in the correct numeric type.
    Expr half         = cast<num_t>(Expr(0.5));
    Expr seven_tenths = cast<num_t>(Expr(0.7));

    // Derived extents (as Expr) from the input ImageParams.
    // We use these instead of compile-time NX/NY so the pipeline remains generic.
    Expr ny_expr = hz_param.dim(0).extent();  // width  (C's ny), mapped to j
    Expr nx_expr = hz_param.dim(1).extent();  // height (C's nx), mapped to i

    // -------------------------------------------------------------------------
    // Optimized single-timestep pipeline, expressed as pure Funcs
    // -------------------------------------------------------------------------
    //
    // We deliberately avoid update definitions here and instead write each
    // field as a single pure expression, using select() to handle the
    // boundary cases. This lets Halide generate a single pass per Func with
    // straightforward, vectorizable loops and makes scheduling much easier.
    //
    // To keep Halide's bounds inference happy, every potentially out-of-bounds
    // access is guarded by an explicit clamp() to the valid domain, so we
    // never index ImageParams outside their declared bounds.
    //

    // Safe wrapper around hz_param with clamped coordinates, so that any
    // use of (j-1) or (i-1) cannot go out-of-bounds.
    Func hz_clamped("hz_clamped");
    hz_clamped(j, i) = hz_param(
        clamp(j, 0, ny_expr - 1),
        clamp(i, 0, nx_expr - 1));

    // New EY after one time-step.
    Func ey1("ey1");

    // C code:
    //   for (j = 0; j < ny; j++)
    //     ey[0][j] = _fict_[t];
    //   for (i = 1; i < nx; i++)
    //     for (j = 0; j < ny; j++)
    //       ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]);
    //
    // Halide layout: (j, i) == C's [i][j].
    //
    // So:
    //   ey1(j, 0)       = fict_param
    //   ey1(j, i>0)     = ey_old(j, i) - 0.5 * (hz_old(j, i) - hz_old(j, i-1))
    //
    // We express this as a single pure Func using select on i==0.
    ey1(j, i) =
        select(i == 0,
               fict_param,
               ey_param(j, i) -
                   half * (hz_clamped(j, i) - hz_clamped(j, i - 1)));

    // New EX after one time-step.
    Func ex1("ex1");

    // C code:
    //   for (i = 0; i < nx; i++)
    //     for (j = 1; j < ny; j++)
    //       ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
    //
    // So:
    //   ex1(0, i)       = ex_old(0, i)
    //   ex1(j>0, i)     = ex_old(j, i) - 0.5 * (hz_old(j, i) - hz_old(j-1, i))
    ex1(j, i) =
        select(j == 0,
               ex_param(j, i),
               ex_param(j, i) -
                   half * (hz_clamped(j, i) - hz_clamped(j - 1, i)));

    // To safely access ex1(j+1, i) and ey1(j, i+1) when computing hz1,
    // we create clamped wrappers around ex1 and ey1. These are only
    // used at interior points, but the clamping guarantees no OOB
    // access appears in the IR even near the boundaries.
    Func ex1_clamped("ex1_clamped"), ey1_clamped("ey1_clamped");
    ex1_clamped(j, i) = ex1(
        clamp(j, 0, ny_expr - 1),
        clamp(i, 0, nx_expr - 1));
    ey1_clamped(j, i) = ey1(
        clamp(j, 0, ny_expr - 1),
        clamp(i, 0, nx_expr - 1));

    // New HZ after one time-step.
    Func hz1("hz1");

    // C code:
    //   for (i = 0; i < nx - 1; i++)
    //     for (j = 0; j < ny - 1; j++)
    //       hz[i][j] = hz[i][j]
    //                  - 0.7 * (ex[i][j+1] - ex[i][j]
    //                          + ey[i+1][j] - ey[i][j]);
    //
    // i.e. for interior (i < nx-1 && j < ny-1) we apply the stencil
    // using *updated* ex/ey; otherwise hz stays unchanged.
    Expr interior = (i < nx_expr - 1) && (j < ny_expr - 1);
    hz1(j, i) =
        select(interior,
               hz_param(j, i) -
                   seven_tenths *
                       ((ex1_clamped(j + 1, i) - ex1_clamped(j, i)) +
                        (ey1_clamped(j, i + 1) - ey1_clamped(j, i))),
               hz_param(j, i));

    // -------------------------------------------------------------------------
    // Scheduling
    // -------------------------------------------------------------------------
    //
    // We now add an explicit schedule to improve performance:
    //  - Compute each Func at the root (one pass per field per time-step).
    //  - Parallelize across rows (i) to use all CPU cores.
    //  - Vectorize across columns (j) using the natural vector width.
    //
    // This is a conservative, cache-friendly schedule that preserves the
    // original dependence structure (ex/ey from old hz; hz from new ex/ey).
    //

    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Compute EY^t+1
    ey1.compute_root()
       .parallel(i)
       .vectorize(j, vec_width);

    // Compute EX^t+1
    ex1.compute_root()
       .parallel(i)
       .vectorize(j, vec_width);

    // Compute HZ^t+1 (depends on ex1 and ey1)
    hz1.compute_root()
       .parallel(i)
       .vectorize(j, vec_width);

    // Build a Pipeline with three outputs: ex1, ey1, hz1.
    Pipeline fdtd_step({ex1, ey1, hz1});

    // JIT-compile once with the chosen target; we'll re-use it every time-step.
    fdtd_step.compile_jit(target);

    // Double-buffering pointers for time-stepping.
    Halide::Buffer<num_t, 2> *ex_curr = &exA;
    Halide::Buffer<num_t, 2> *ey_curr = &eyA;
    Halide::Buffer<num_t, 2> *hz_curr = &hzA;

    Halide::Buffer<num_t, 2> *ex_next = &exB;
    Halide::Buffer<num_t, 2> *ey_next = &eyB;
    Halide::Buffer<num_t, 2> *hz_next = &hzB;

    // Start timing.
    auto start = std::chrono::high_resolution_clock::now();

    // Time-stepping loop (host-driven).
    for (int t = 0; t < tmax; t++) {
        // Bind current input fields.
        ex_param.set(*ex_curr);
        ey_param.set(*ey_curr);
        hz_param.set(*hz_curr);

        // Set current _fict_[t] value.
        fict_param.set(fict(t));

        // Realize next time-step into ex_next, ey_next, hz_next.
        Realization r({*ex_next, *ey_next, *hz_next});
        fdtd_step.realize(r);

        // Swap current and next buffers for the next iteration.
        std::swap(ex_curr, ex_next);
        std::swap(ey_curr, ey_next);
        std::swap(hz_curr, hz_next);
    }

    // Stop timing.
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Final fields are in *ex_curr, *ey_curr, *hz_curr.
    Halide::Buffer<num_t, 2> &ex_final = *ex_curr;
    Halide::Buffer<num_t, 2> &ey_final = *ey_curr;
    Halide::Buffer<num_t, 2> &hz_final = *hz_curr;

    // Optional: print the arrays to prevent dead-code elimination
    // (and to roughly match the original PolyBench behavior).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);

        // Print ex
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < ny; jj++) {
                std::cerr << ex_final(jj, ii) << '\n';
            }
        }

        // Print ey
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < ny; jj++) {
                std::cerr << ey_final(jj, ii) << '\n';
            }
        }

        // Print hz
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < ny; jj++) {
                std::cerr << hz_final(jj, ii) << '\n';
            }
        }
    }

    // Print elapsed time in seconds (as in gemm.cpp example).
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}