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

// Halide-based initialization of A and B.
//
// This version adds an explicit schedule that
//  - vectorizes over the natural SIMD width of the host
//  - parallelizes across chunks of the 1D domain
// while preserving the original initialization formula.
static void init_array(int n,
                       Halide::Buffer<num_t, 1> A,
                       Halide::Buffer<num_t, 1> B) {
    Var i("i");
    Func init_A("init_A"), init_B("init_B");

    // A[i] = ((DATA_TYPE) i + 2) / n;
    init_A(i) = cast<num_t>(i + 2) / cast<num_t>(n);

    // B[i] = ((DATA_TYPE) i + 3) / n;
    init_B(i) = cast<num_t>(i + 3) / cast<num_t>(n);

    // Choose a vector width based on the host target and num_t.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Chunk size for parallelism: a few vectors per chunk to keep
    // threads busy but not too small to amortize overhead.
    const int chunk_size = vec_width * 32;

    // Split i into a parallel outer chunk and a vectorized inner lane.
    Var io("io"), ii("ii");

    init_A
        .compute_root()
        .split(i, io, ii, chunk_size)
        .parallel(io)
        .vectorize(ii, vec_width);

    init_B
        .compute_root()
        .split(i, io, ii, chunk_size)
        .parallel(io)
        .vectorize(ii, vec_width);

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

    // Initialize A and B.
    init_array(n, A, B);

    // ImageParams used as inputs to the per-time-step Halide kernels.
    ImageParam A_param(type_of<num_t>(), 1, "A_param");
    ImageParam B_param(type_of<num_t>(), 1, "B_param");

    // Var for the interior index (0 .. n-3), corresponding to i = k+1 in C.
    Var k("k");

    // Center index in the original array: i = k + 1 (i runs from 1 .. n-2).
    Expr i_center = k + 1;

    // Constant factor 0.33333 cast to num_t.
    Expr coeff = cast<num_t>(Expr(0.33333));

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

    // --------------------------------------------------------------------
    // Optimized schedules for the Jacobi sweeps
    //
    // We:
    //  - Use the host's natural vector width for num_t.
    //  - Split the 1D domain into chunks and parallelize across them.
    //  - Vectorize within each chunk for good SIMD utilization.
    //
    // Data locality is already optimal for a 1D stencil since accesses
    // are to consecutive elements; these schedules just expose more
    // parallelism and SIMD opportunities while preserving semantics.
    // --------------------------------------------------------------------
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());
    // Chunk size for parallelism: several vectors worth of work.
    const int chunk_size = vec_width * 64;

    Var ko("ko"), ki("ki");

    jacobi_A_to_B
        .compute_root()
        .split(k, ko, ki, chunk_size)
        .parallel(ko)
        .vectorize(ki, vec_width);

    jacobi_B_to_A
        .compute_root()
        .split(k, ko, ki, chunk_size)
        .parallel(ko)
        .vectorize(ki, vec_width);

    // Bind the interior [1 .. n-2] of A and B to contiguous Halide Buffers.
    // k runs from 0 .. n-3; k == 0 corresponds to element index 1.
    Halide::Buffer<num_t, 1> A_interior(A.data() + 1, n - 2);
    Halide::Buffer<num_t, 1> B_interior(B.data() + 1, n - 2);

    // Connect ImageParams to the full arrays once. The underlying buffers
    // keep the same base pointer; we only overwrite their contents.
    A_param.set(A);
    B_param.set(B);

    // JIT-compile the per-sweep kernels once for the chosen target.
    jacobi_A_to_B.compile_jit(target);
    jacobi_B_to_A.compile_jit(target);

    // Time the Jacobi iterations (the main kernel).
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < tsteps; t++) {
        // First half-step: A -> B (update B[1..n-2] from A).
        jacobi_A_to_B.realize(B_interior);

        // Second half-step: B -> A (update A[1..n-2] from B).
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