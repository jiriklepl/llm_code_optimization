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
    // i, j, k here are the C-style indices; buffer indices are (k, j, i).
    // Use num_t literals to get correct Halide constant type.
    Expr val = cast<num_t>(i + j + (n - k)) * (num_t)10 / (num_t)n;
    init(k, j, i) = val;

    // For initialization, the default schedule is fine. It's executed once
    // and is not in the timed region, so we keep it simple.
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

    // Create ImageParams for A and B so that Halide pipelines can read them.
    ImageParam A_param(type_of<num_t>(), 3, "A_param");
    ImageParam B_param(type_of<num_t>(), 3, "B_param");

    A_param.set(A);
    B_param.set(B);

    // Vars for 3D indexing (k, j, i) <-> (C's k, j, i).
    Var k("k"), j("j"), i("i");

    // One time-step consists of:
    //   1) B = stencil(A)
    //   2) A = stencil(B)
    //
    // We'll express these as two separate Halide Funcs:
    //   B_from_A: reads A_param, writes into B
    //   A_from_B: reads B_param, writes into A
    //
    // We only update interior points (i,j,k in [1, n-2]), as in the
    // original C loops. The boundaries remain unchanged.
    Func B_from_A("B_from_A");
    Func A_from_B("A_from_B");

    // Stencil coefficients; use num_t so Halide Expr has the right type.
    Expr alpha = (num_t)0.125;
    Expr two   = (num_t)2.0;

    // --- Stage 1: B = stencil(A) ---
    {
        Expr centerA = A_param(k, j, i);

        Expr term_i = alpha *
                      (A_param(k, j, i + 1) - two * centerA + A_param(k, j, i - 1));
        Expr term_j = alpha *
                      (A_param(k, j + 1, i) - two * centerA + A_param(k, j - 1, i));
        Expr term_k = alpha *
                      (A_param(k + 1, j, i) - two * centerA + A_param(k - 1, j, i));

        B_from_A(k, j, i) = term_i + term_j + term_k + centerA;
    }

    // --- Stage 2: A = stencil(B) ---
    {
        Expr centerB = B_param(k, j, i);

        Expr term_i = alpha *
                      (B_param(k, j, i + 1) - two * centerB + B_param(k, j, i - 1));
        Expr term_j = alpha *
                      (B_param(k, j + 1, i) - two * centerB + B_param(k, j - 1, i));
        Expr term_k = alpha *
                      (B_param(k + 1, j, i) - two * centerB + B_param(k - 1, j, i));

        A_from_B(k, j, i) = term_i + term_j + term_k + centerB;
    }

    // -----------------------------
    // Scheduling for performance
    // -----------------------------
    //
    // Memory layout is (k, j, i) with k contiguous (dim 0), then j (dim 1),
    // then i (dim 2). We:
    //   * keep k as innermost loop for unit-stride accesses;
    //   * tile in (i,j) for cache locality;
    //   * vectorize along k using the hardware natural width;
    //   * parallelize over i-tiles to exploit many cores.
    //
    // This is applied identically to B_from_A and A_from_B.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Tunable tile sizes for i and j (outer spatial dims in C indexing).
    const int tile_i = 8;
    const int tile_j = 8;

    Var io("io"), jo("jo"), ii("ii"), ji("ji");

    // B_from_A: compute_root so that we can tile/parallelize the whole domain.
    B_from_A
        .compute_root()
        // Tile (i,j) to improve cache locality.
        // Note: i and j here are the C-style indices; Halide storage order
        // is still (k, j, i) with k contiguous in memory.
        .tile(i, j, io, jo, ii, ji, tile_i, tile_j)
        // Reorder so that:
        //   outermost loops: io (i tiles), then jo (j tiles)
        //   then ii, ji inside the tile
        //   innermost loop: k (contiguous in memory)
        //
        // reordering arguments from innermost to outermost.
        .reorder(k, ji, ii, jo, io)
        // Vectorize along contiguous k dimension.
        .vectorize(k, vec_width)
        // Parallelize across tiles in the i-dimension (io).
        // This exposes coarse-grain parallelism over the 3D domain.
        .parallel(io);

    // A_from_B: identical schedule for symmetry and locality.
    A_from_B
        .compute_root()
        .tile(i, j, io, jo, ii, ji, tile_i, tile_j)
        .reorder(k, ji, ii, jo, io)
        .vectorize(k, vec_width)
        .parallel(io);

    // -----------------------------
    // Interior views for updates
    // -----------------------------
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

    // Compile the kernels ahead of time to avoid measuring JIT cost.
    B_from_A.compile_jit(target);
    A_from_B.compile_jit(target);

    // Time the tsteps iterations.
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < tsteps; t++) {
        // First half-step: B = stencil(A) on the interior.
        B_from_A.realize(B_interior);

        // Second half-step: A = stencil(B) on the interior.
        A_from_B.realize(A_interior);
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