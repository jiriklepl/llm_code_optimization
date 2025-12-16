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
    Expr half          = cast<num_t>(Expr(0.5));
    Expr seven_tenths  = cast<num_t>(Expr(0.7));

    // New fields after a single time step.
    Func ey1("ey1"), ex1("ex1"), hz1("hz1");

    // Base (pure) definitions: start with the old values.
    ey1(j, i) = ey_param(j, i);
    ex1(j, i) = ex_param(j, i);
    hz1(j, i) = hz_param(j, i);

    // Now add update definitions that correspond to the C loops.

    // C code:
    // for (j = 0; j < ny; j++)
    //   ey[0][j] = _fict_[t];
    //
    // In Halide: i == 0 row, j runs over all columns.
    {
        RDom r_ey0(0, ey_param.dim(0).extent(), "r_ey0");
        ey1(r_ey0.x, 0) = fict_param;
    }

    // C code:
    // for (i = 1; i < nx; i++)
    //   for (j = 0; j < ny; j++)
    //     ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]);
    //
    // Halide: i in [1..nx-1], j in [0..ny-1]
    {
        RDom r_ey1(0, ey_param.dim(0).extent(),      // j: 0 .. ny-1
                   1, ey_param.dim(1).extent() - 1,  // i: 1 .. nx-1
                   "r_ey1");

        ey1(r_ey1.x, r_ey1.y) =
            ey1(r_ey1.x, r_ey1.y) -
            half * (hz_param(r_ey1.x, r_ey1.y) - hz_param(r_ey1.x, r_ey1.y - 1));
    }

    // C code:
    // for (i = 0; i < nx; i++)
    //   for (j = 1; j < ny; j++)
    //     ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
    //
    // Halide: i in [0..nx-1], j in [1..ny-1]
    {
        RDom r_ex(1, ex_param.dim(0).extent() - 1,   // j: 1 .. ny-1
                  0, ex_param.dim(1).extent(),       // i: 0 .. nx-1
                  "r_ex");

        ex1(r_ex.x, r_ex.y) =
            ex1(r_ex.x, r_ex.y) -
            half * (hz_param(r_ex.x, r_ex.y) - hz_param(r_ex.x - 1, r_ex.y));
    }

    // C code:
    // for (i = 0; i < nx - 1; i++)
    //   for (j = 0; j < ny - 1; j++)
    //     hz[i][j] = hz[i][j]
    //                - 0.7 * (ex[i][j+1] - ex[i][j]
    //                        + ey[i+1][j] - ey[i][j]);
    //
    // Halide: i in [0..nx-2], j in [0..ny-2]
    {
        RDom r_hz(0, hz_param.dim(0).extent() - 1,   // j: 0 .. ny-2
                  0, hz_param.dim(1).extent() - 1,   // i: 0 .. nx-2
                  "r_hz");

        hz1(r_hz.x, r_hz.y) =
            hz1(r_hz.x, r_hz.y) -
            seven_tenths *
            ((ex1(r_hz.x + 1, r_hz.y) - ex1(r_hz.x, r_hz.y)) +
             (ey1(r_hz.x, r_hz.y + 1) - ey1(r_hz.x, r_hz.y)));
    }

    // Build a Pipeline with three outputs: ex1, ey1, hz1.
    Pipeline fdtd_step({ex1, ey1, hz1});

    // JIT-compile once; we'll re-use it for every time-step.
    fdtd_step.compile_jit();

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
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                std::cerr << ex_final(j, i) << '\n';
            }
        }

        // Print ey
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                std::cerr << ey_final(j, i) << '\n';
            }
        }

        // Print hz
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                std::cerr << hz_final(j, i) << '\n';
            }
        }
    }

    // Print elapsed time in seconds (as in gemm.cpp example).
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}