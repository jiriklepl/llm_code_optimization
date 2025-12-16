#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// Common PolyBench-style definitions (DATA_TYPE, N, TSTEPS, etc.)
#include "defines.hpp"
// Benchmark-specific definitions for jacobi-1d (N, TSTEPS, etc.)
#include "jacobi-1d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// Helper: pick a reasonable SIMD width for the current target and element type.
static int pick_vector_width() {
    Target target = get_jit_target_from_environment();
    return natural_vector_size(target, type_of<num_t>());
}

// Halide-based initialization of A and B.
static void init_array(int n,
                       Halide::Buffer<num_t, 1> A,
                       Halide::Buffer<num_t, 1> B) {
    Var i("i");
    Func init_A("init_A"), init_B("init_B");

    // A[i] = ((DATA_TYPE) i + 2) / n;
    init_A(i) = (cast<num_t>(i) + cast<num_t>(2)) / cast<num_t>(n);

    // B[i] = ((DATA_TYPE) i + 3) / n;
    init_B(i) = (cast<num_t>(i) + cast<num_t>(3)) / cast<num_t>(n);

    // ------------------------------------------------------------------
    // Schedule for initialization:
    // - Split the 1D domain of i into tiles to get coarse-grain parallelism.
    // - Vectorize within each tile to exploit SIMD.
    // This is overkill for very small n, but for large N it gives
    // good CPU utilization and cache behavior.
    // ------------------------------------------------------------------
    const int tile_size   = 1024;                 // spatial tile size (tunable)
    const int vector_width = pick_vector_width(); // SIMD width in elements

    Var io("io"), ii("ii");

    init_A
        .compute_root()
        .split(i, io, ii, tile_size)  // i = io * tile_size + ii
        .vectorize(ii, vector_width)  // SIMD over contiguous ii
        .parallel(io);                // tiles in parallel

    init_B
        .compute_root()
        .split(i, io, ii, tile_size)
        .vectorize(ii, vector_width)
        .parallel(io);

    // Realize into the provided buffers.
    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem size.
    int n      = N;
    int tsteps = TSTEPS;

    // Allocate 1D buffers A and B of length n.
    Halide::Buffer<num_t, 1> A(n);
    Halide::Buffer<num_t, 1> B(n);

    // Initialize A and B (with a scheduled Halide pipeline).
    init_array(n, A, B);

    // ImageParams used as inputs to the per-time-step Halide kernels.
    ImageParam A_param(type_of<num_t>(), 1, "A_param");
    ImageParam B_param(type_of<num_t>(), 1, "B_param");

    // Var for the interior index (0 .. n-3), corresponding to i = k+1 in C.
    Var k("k");

    // Center index in the original array: i = k + 1 (i runs from 1 .. n-2).
    Expr i_center = k + 1;

    // Constant factor 0.33333 cast to num_t.
    Expr coeff = cast<num_t>(0.33333f);

    // One Jacobi sweep: A -> B (compute B[i] from neighbors of A[i]).
    Func jacobi_A_to_B("jacobi_A_to_B");
    jacobi_A_to_B(k) =
        coeff * (A_param(i_center - 1) +
                 A_param(i_center) +
                 A_param(i_center + 1));

    // One Jacobi sweep: B -> A (compute A[i] from neighbors of B[i]).
    Func jacobi_B_to_A("jacobi_B_to_A");
    jacobi_B_to_A(k) =
        coeff * (B_param(i_center - 1) +
                 B_param(i_center) +
                 B_param(i_center + 1));

    // ------------------------------------------------------------------
    // Optimized schedules for the two sweeps.
    //
    // For a fixed time step t:
    //   - jacobi_A_to_B reads only from A and writes distinct B[k+1]
    //   - jacobi_B_to_A reads only from B and writes distinct A[k+1]
    // so there are no cross-k dependencies within a sweep.
    //
    // We therefore:
    //   * split k into tiles of size tile_k
    //   * parallelize across tiles
    //   * vectorize within each tile
    //
    // This matches the optimization model: spatial blocking + parallelize(I)
    // + vectorize(inner).
    // ------------------------------------------------------------------
    const int tile_k       = 1024;                // tile size in k (tunable)
    const int vector_width = pick_vector_width(); // SIMD width in elements

    Var ko("ko"), ki("ki");

    jacobi_A_to_B
        .compute_root()
        .split(k, ko, ki, tile_k)     // k = ko * tile_k + ki
        .vectorize(ki, vector_width)  // SIMD on contiguous interior points
        .parallel(ko);                // tiles in parallel

    jacobi_B_to_A
        .compute_root()
        .split(k, ko, ki, tile_k)
        .vectorize(ki, vector_width)
        .parallel(ko);

    // Bind the interior [1 .. n-2] of A and B to contiguous Halide Buffers.
    // k runs from 0 .. n-3; k == 0 corresponds to element index 1.
    Halide::Buffer<num_t, 1> A_interior(A.data() + 1, n - 2);
    Halide::Buffer<num_t, 1> B_interior(B.data() + 1, n - 2);

    // JIT-compile the per-sweep kernels once for the current target.
    Target target = get_jit_target_from_environment();
    jacobi_A_to_B.compile_jit(target);
    jacobi_B_to_A.compile_jit(target);

    // Time the Jacobi iterations (the main kernel).
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < tsteps; t++) {
        // First half-step: A -> B (update B[1..n-2] from A).
        A_param.set(A);
        jacobi_A_to_B.realize(B_interior);

        // Second half-step: B -> A (update A[1..n-2] from B).
        B_param.set(B);
        jacobi_B_to_A.realize(A_interior);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the output array A to stderr (to mimic PolyBench DCE).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(6);
        for (int i = 0; i < n; i++) {
            std::cerr << A(i) << '\n';
        }
    }

    // Print the elapsed time (in seconds) to stdout.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}