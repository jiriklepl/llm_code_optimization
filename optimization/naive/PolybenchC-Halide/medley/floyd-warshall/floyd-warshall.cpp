#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

#include "Halide.h"

// include common PolyBench-style definitions
#include "defines.hpp"

// include benchmark-specific definitions (defines N and DATA_TYPE)
#include "floyd-warshall.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize the path matrix.
 *
 * Original C:
 *   for (i = 0; i < n; i++)
 *     for (j = 0; j < n; j++) {
 *       path[i][j] = i*j%7+1;
 *       if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
 *          path[i][j] = 999;
 *     }
 *
 * C array path[i][j] is mapped to Buffer(j, i).
 */
static void init_array(int n, Buffer<num_t, 2> path) {
    Var i("i"), j("j");
    Func init_path("init_path");

    // Base value: i * j % 7 + 1
    Expr base = cast<num_t>((i * j % 7) + 1);

    // Condition for setting to 999
    Expr sum = i + j;
    Expr cond = ((sum % 13) == 0) ||
                ((sum % 7)  == 0) ||
                ((sum % 11) == 0);

    // Buffer layout is (x = j, y = i), so we write init_path(j, i).
    init_path(j, i) = select(cond,
                             cast<num_t>(999),
                             base);

    // ------------------------------------------------------------------
    // Schedule for initialization:
    //  - j is the contiguous dimension (row-major in C: path[i][j])
    //  - vectorize across j for SIMD
    //  - parallelize across i (rows)
    // ------------------------------------------------------------------
    Target t = get_jit_target_from_environment();
    int vec_width = t.natural_vector_size(type_of<num_t>());
    if (vec_width < 1) {
        vec_width = 1;
    }

    init_path
        .reorder(j, i)          // make j innermost, i outermost
        .vectorize(j, vec_width)
        .parallel(i);

    // Realize into the provided buffer. Buffer layout is (x = j, y = i).
    init_path.realize(path);
}

/**
 * Print the path matrix (mainly to prevent dead-code elimination
 * and for optional correctness checking).
 *
 * Mirrors the PolyBench print_array semantics, but prints to stderr.
 */
static void print_array(int n, Buffer<num_t, 2> path) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // path[i][j] -> path(j, i)
            std::cerr << path(j, i) << '\n';
        }
    }
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Buffers corresponding to: DATA_TYPE path[N][N]
    // We use two buffers for double-buffered Floyd–Warshall.
    Buffer<num_t, 2> path0(n, n);
    Buffer<num_t, 2> path1(n, n);

    // Initialize data into path0
    init_array(n, path0);

    // Halide parameters and pipeline for a single Floyd–Warshall step.
    //
    // We follow the recommended DP translation pattern:
    //   For each k:
    //     step(i,j) = min(path(i,j), path(i,k) + path(k,j))
    //   implemented as a Func that reads from an ImageParam (current path)
    //   and a scalar Param k, and then executed from a host-side loop
    //   with double-buffering.
    ImageParam path_param(type_of<num_t>(), 2, "path_param");
    Param<int> k_param("k");

    Var i("i"), j("j");
    Func floyd_step("floyd_step");

    // Mapping:
    //   C: path[i][j] -> path_param(j, i)
    //   C: path[i][k] -> path_param(k, i)
    //   C: path[k][j] -> path_param(j, k)
    Expr current = path_param(j, i);
    Expr via_k   = path_param(k_param, i) + path_param(j, k_param);

    floyd_step(j, i) = Halide::min(current, via_k);

    // ------------------------------------------------------------------
    // Schedule for the Floyd–Warshall step:
    //
    //  - We keep j as the innermost loop to match the buffer layout
    //    (x = j is contiguous in memory).
    //  - Vectorize across j for SIMD.
    //  - Parallelize across i (rows) since each (i,j) for a fixed k
    //    is independent.
    //
    //  This preserves the original semantics (same per-(i,j,k) update)
    //  while improving memory throughput and CPU utilization.
    // ------------------------------------------------------------------
    Target t = get_jit_target_from_environment();
    int vec_width = t.natural_vector_size(type_of<num_t>());
    if (vec_width < 1) {
        vec_width = 1;
    }

    floyd_step
        .reorder(j, i)              // j inner, i outer
        .parallel(i)                // multi-thread over rows
        .vectorize(j, vec_width);   // SIMD across columns

    // Bind initial data and compile the step once.
    path_param.set(path0);
    k_param.set(0); // will be updated inside the loop
    floyd_step.compile_jit(t);

    // Double-buffering pointers
    Buffer<num_t, 2> *current_path = &path0;
    Buffer<num_t, 2> *next_path    = &path1;

    // Time only the Floyd–Warshall kernel (not initialization).
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        // Bind the current path and the iteration index k
        path_param.set(*current_path);
        k_param.set(k);

        // Compute the next path
        floyd_step.realize(*next_path);

        // Swap buffers for the next iteration
        std::swap(current_path, next_path);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // The final result is in *current_path
    Buffer<num_t, 2> &path = *current_path;

    // Optional printing to prevent dead-code elimination and for debugging.
    // This mirrors the style from example/gemm.cpp.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        print_array(n, path);
    }

    // Print kernel execution time in seconds
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}