#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions (DATA_TYPE, N, TSTEPS, SCALAR_VAL, etc)
#include "defines.hpp"

// include benchmark-specific definitions for seidel-2d
#include "seidel-2d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize the array A with the same values as the original C code:
 *
 *   for (i = 0; i < n; i++)
 *     for (j = 0; j < n; j++)
 *       A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
 *
 * Note: C uses A[i][j] (row-major). Halide uses Buffer(x, y) with x as the
 * fastest-varying dimension; we map A[i][j] -> A(j, i).
 */
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    Var i("i"), j("j");
    Func init_A("init_A");

    // A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
    // Map to Halide as A(j, i).
    init_A(j, i) = cast<num_t>(i * (j + 2) + 2) / cast<num_t>(n);

    init_A.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size and time steps
    int n = N;
    int tsteps = TSTEPS;

    // Buffers for current and next time-step values.
    // We treat dimension 0 as column (j), dimension 1 as row (i),
    // so that C's A[i][j] maps to A(j, i).
    Buffer<num_t, 2> A_cur(n, n);
    Buffer<num_t, 2> A_next(n, n);

    // Initialize A_cur
    init_array(n, A_cur);

    // ImageParam to represent the "previous" time-step array in Halide
    ImageParam A_prev_param(type_of<num_t>(), 2, "A_prev");

    // Vars for spatial indices
    Var x("x"), y("y");

    // One time-step of the 2D Seidel/Jacobi-like update, expressed in Halide.
    //
    // Original C update (for interior points 1..n-2):
    //
    // A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
    //            + A[i][j-1] + A[i][j] + A[i][j+1]
    //            + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0;
    //
    // Mapping to Halide coordinates: i -> y, j -> x, A[i][j] -> A_prev_param(x, y).
    Func seidel_step("seidel_step");

    // Strength-reduce the division by 9.0 into a multiplication by a
    // precomputed reciprocal. This matches the mathematical operation but
    // avoids a division in the inner loop on most targets.
    Expr inv_denom = cast<num_t>(SCALAR_VAL(1.0) / SCALAR_VAL(9.0));

    seidel_step(x, y) =
        (A_prev_param(x - 1, y - 1) +
         A_prev_param(x - 1, y) +
         A_prev_param(x - 1, y + 1) +
         A_prev_param(x,     y - 1) +
         A_prev_param(x,     y) +
         A_prev_param(x,     y + 1) +
         A_prev_param(x + 1, y - 1) +
         A_prev_param(x + 1, y) +
         A_prev_param(x + 1, y + 1)) * inv_denom;

    // --------------------------------------------------------------------
    // Schedule for seidel_step
    //
    // We keep the algorithm exactly the same, but improve performance by:
    //   - parallelizing across rows (y),
    //   - vectorizing across columns (x) using the natural SIMD width
    //     for this target.
    //
    // The time dimension (t) is kept outside Halide in a C++ loop,
    // which is the recommended way to express loop-carried dependences.
    // --------------------------------------------------------------------
    Target target = get_jit_target_from_environment();

    int vec_width = target.natural_vector_size(type_of<num_t>());
    if (vec_width < 1) {
        vec_width = 1;  // Fallback in case the target reports no vector width.
    }

    // Parallelize over rows (y): each worker processes a set of rows.
    // This preserves the original row-major traversal and exposes
    // coarse-grain parallelism across the 2D grid.
    //
    // Vectorize over columns (x): x is the fastest-varying dimension
    // in memory (Buffer(x, y)), so this generates contiguous SIMD
    // loads/stores and good cache utilization.
    seidel_step
        .parallel(y)
        .vectorize(x, vec_width);

    // JIT-compile the per-time-step pipeline once for the chosen target.
    seidel_step.compile_jit(target);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Double-buffering over time steps.
    // We keep pointers to the "current" and "next" buffers and swap each step.
    Buffer<num_t, 2> *A_src = &A_cur;
    Buffer<num_t, 2> *A_dst = &A_next;

    for (int t = 0; t < tsteps; t++) {
        // Bind the current state as the input image for this step
        A_prev_param.set(*A_src);

        // Realize only the interior region [1 .. n-2] x [1 .. n-2].
        // This keeps all stencil accesses in-bounds without explicit
        // boundary conditions, since neighbors of interior points are
        // always in [0 .. n-1].
        Buffer<num_t> interior =
            A_dst->cropped(0, 1, n - 2) // crop x from 1, extent n-2 -> [1..n-2]
                 .cropped(1, 1, n - 2); // crop y from 1, extent n-2 -> [1..n-2]

        seidel_step.realize(interior);

        // Copy boundary values (edges) unchanged from the previous state.
        // In the original C code, boundaries are never updated (loops start at 1
        // and end at n-2), so they remain constant across time steps.
        //
        // Top and bottom rows
        for (int j = 0; j < n; j++) {
            (*A_dst)(j, 0)     = (*A_src)(j, 0);
            (*A_dst)(j, n - 1) = (*A_src)(j, n - 1);
        }
        // Left and right columns (excluding corners which are already set)
        for (int i = 1; i < n - 1; i++) {
            (*A_dst)(0,     i) = (*A_src)(0,     i);
            (*A_dst)(n - 1, i) = (*A_src)(n - 1, i);
        }

        // Swap the role of the buffers for the next iteration
        Buffer<num_t, 2> *tmp = A_src;
        A_src = A_dst;
        A_dst = tmp;
    }

    // After the loop, A_src points to the buffer with the final data
    Buffer<num_t, 2> &A_final = *A_src;

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally dump the result to stderr (similar to the gemm example)
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Recall: A[i][j] in C is A_final(j, i) in Halide.
                std::cerr << A_final(j, i) << '\n';
            }
        }
    }

    // Print elapsed time in seconds with 6 decimal places
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}