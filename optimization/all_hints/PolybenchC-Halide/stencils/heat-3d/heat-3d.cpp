#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// Include common PolyBench-style definitions (DATA_TYPE, SCALAR_VAL, N, TSTEPS, etc.)
#include "defines.hpp"

// Include benchmark-specific definitions (N, TSTEPS for heat-3d)
#include "heat-3d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// -----------------------------------------------------------------------------
// Helper: choose a reasonable vector width for the current CPU and element type
// -----------------------------------------------------------------------------
static int vector_width_for_target(const Halide::Target &target) {
    // Choose vector register width in bytes based on available ISA.
    int vector_bytes = 0;
    if (target.has_feature(Target::AVX512_Skylake) ||
        target.has_feature(Target::AVX512_KNL) ||
        target.has_feature(Target::AVX512)) {
        vector_bytes = 64;  // 512-bit vectors
    } else if (target.has_feature(Target::AVX2) ||
               target.has_feature(Target::AVX)) {
        vector_bytes = 32;  // 256-bit vectors
    } else if (target.has_feature(Target::SSE41) ||
               target.has_feature(Target::SSE2)) {
        vector_bytes = 16;  // 128-bit vectors
    } else {
        // Fallback: no SIMD. Use scalar code.
        vector_bytes = static_cast<int>(sizeof(num_t));
    }

    int w = std::max(1, vector_bytes / static_cast<int>(sizeof(num_t)));
    return w;
}

/**
 * Initialize A and B as in the original C version:
 *
 *   A[i][j][k] = B[i][j][k] = (i + j + (n - k)) * 10 / n;
 *
 * Mapping from C arrays to Halide buffers:
 *   C: DATA_TYPE A[N][N][N] accessed as A[i][j][k]
 *   Halide: Buffer<num_t, 3> A(k, j, i)  // dim 0 = k, dim 1 = j, dim 2 = i
 *
 * So A[i][j][k] in C becomes A(k, j, i) in Halide.
 */
static void init_array(int n,
                       Halide::Buffer<num_t, 3> A,
                       Halide::Buffer<num_t, 3> B) {
    Var k("k"), j("j"), i("i");

    Func init("init");
    // i, j, k here are the C-style indices; buffer indices are (k, j, i)
    Expr val =
        cast<num_t>(i + j + (n - k)) * cast<num_t>(10) / cast<num_t>(n);
    init(k, j, i) = val;

    // Simple CPU schedule: row-major over (i,j,k) with k innermost.
    Target target = get_jit_target_from_environment();
    const int vec = vector_width_for_target(target);

    init
        .compute_root()
        // Ensure k is the innermost loop, then j, then i.
        .reorder(k, j, i)
        // Vectorize across contiguous k.
        .vectorize(k, vec)
        // Parallelize across the outermost i dimension (z in C layout).
        .parallel(i);

    // Compile once, then realize twice into A and B (B initially equals A).
    init.compile_jit(target);
    init.realize(A);
    init.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem size and number of time steps.
    int n      = N;
    int tsteps = TSTEPS;

    // Allocate 3D buffers.
    // Dimension order is (k, j, i) so that:
    //   A(i,j,k in C) <-> A(k, j, i) in Halide.
    Halide::Buffer<num_t, 3> A(n, n, n);
    Halide::Buffer<num_t, 3> B(n, n, n);

    // Initialize A and B.
    init_array(n, A, B);

    // Target and vector width for scheduling.
    Target target = get_jit_target_from_environment();
    const int vec = vector_width_for_target(target);

    // ImageParam representing the current state of the grid.
    // We'll bind it to A or B before each stencil application.
    ImageParam grid_param(type_of<num_t>(), 3, "grid_param");

    // Vars for 3D indexing (k, j, i) <-> (C's k, j, i).
    Var k("k"), j("j"), i("i");

    // One time-step consists of:
    //   1) B = stencil(A)
    //   2) A = stencil(B)
    //
    // We'll express a *single* generic stencil Func:
    //   heat_step: reads from grid_param, writes into either A or B.
    Func heat_step("heat_step");

    // Constants used in the stencil.
    const Expr alpha = cast<num_t>(Expr(0.125));  // diffusion coefficient
    const Expr six   = cast<num_t>(Expr(6.0));

    // Center value.
    Expr center = grid_param(k, j, i);

    // 6-point Laplacian (sum of neighbors minus 6*center).
    Expr lap =
        grid_param(k,     j,     i + 1) +
        grid_param(k,     j,     i - 1) +
        grid_param(k,     j + 1, i    ) +
        grid_param(k,     j - 1, i    ) +
        grid_param(k + 1, j,     i    ) +
        grid_param(k - 1, j,     i    ) -
        six * center;

    // Update equation:
    //   new_value = center + alpha * lap;
    // This is algebraically equivalent to the original PolyBench formula
    // but reduces the number of multiplications by alpha and improves
    // common subexpression reuse.
    heat_step(k, j, i) = center + alpha * lap;

    // We want to update only the interior region:
    //   i, j, k in [1, n-2]
    // The boundary values remain unchanged (exactly as in the C code).
    //
    // We'll realize into cropped "interior" views that alias the
    // underlying A and B storage, so that:
    //   - interior points are overwritten by the stencil
    //   - boundary points (index 0 and n-1) are untouched
    Halide::Buffer<num_t, 3> B_interior =
        B.cropped(0, 1, n - 2)
         .cropped(1, 1, n - 2)
         .cropped(2, 1, n - 2);

    Halide::Buffer<num_t, 3> A_interior =
        A.cropped(0, 1, n - 2)
         .cropped(1, 1, n - 2)
         .cropped(2, 1, n - 2);

    // --------------------
    // Schedule for heat_step
    // --------------------
    //
    // Memory layout of A/B is:
    //   dim 0 (k): contiguous (C's k index)
    //   dim 1 (j): stride n
    //   dim 2 (i): stride n*n
    //
    // We:
    //   - Keep k innermost for unit-stride vector loads/stores.
    //   - Parallelize across i (planes) for multicore scalability.
    //   - Let j remain as the middle loop to provide good cache reuse
    //     across rows within each plane.
    heat_step
        .compute_root()
        .reorder(k, j, i)     // k inner, then j, then i outer
        .vectorize(k, vec)    // SIMD across contiguous k dimension
        .parallel(i);         // parallelize across planes of constant i

    // Compile the kernel ahead of time to avoid measuring JIT cost
    // inside the time-stepping loop.
    heat_step.compile_jit(target);

    // Time the tsteps iterations.
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < tsteps; t++) {
        // First half-step: B = stencil(A) on the interior.
        grid_param.set(A);
        heat_step.realize(B_interior);

        // Second half-step: A = stencil(B) on the interior.
        grid_param.set(B);
        heat_step.realize(A_interior);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print out the final A array (similar to print_array in C).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                for (int kk = 0; kk < n; kk++) {
                    // Map C's A[ii][jj][kk] -> A(kk, jj, ii)
                    std::cerr << A(kk, jj, ii) << '\n';
                }
            }
        }
    }

    // Print elapsed time (seconds) to stdout.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}