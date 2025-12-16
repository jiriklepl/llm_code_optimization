#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "lu.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// -----------------------------------------------------------------------------
// Array initialization: translate Polybench init_array() into a Halide pipeline
// -----------------------------------------------------------------------------
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    // We map the original C array A[i][j] to Halide buffer A(j, i):
    //   - i is the row index  -> second dimension (y)
    //   - j is the column index -> first dimension (x)

    Var i("i"), j("j");
    Func A0("A0"), B("B");

    // First, generate the lower-triangular matrix used as in the C code:
    //
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j <= i; j++)
    //     A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
    //   for (j = i+1; j < n; j++)
    //     A[i][j] = 0;
    //   A[i][i] = 1;
    // }
    //
    // Note: for 0 <= j < n, (-j % n) == -j, so we can drop the % n.
    // We fold all this into a single expression:

    Expr one   = cast<num_t>(1);
    Expr zero  = cast<num_t>(0);
    Expr n_val = cast<num_t>(n);

    A0(j, i) = select(
        j <= i,
        // On or below the diagonal
        select(i == j,
               // Diagonal element
               one,
               // Strictly lower triangular part
               cast<num_t>(-j) / n_val + one),
        // Above the diagonal
        zero);

    // Now make the matrix positive semi-definite by computing:
    //
    // B[r][s] = sum_t A[r][t] * A[s][t];
    //
    // and then copying B back into A. In terms of our mapping:
    // B(r, s) in C  -> B(j, i) in Halide
    // A[r][t]       -> A0(t, r) -> A0(t, i)
    // A[s][t]       -> A0(t, s) -> A0(t, j)
    //
    // So:
    //   B(j, i) = sum_{t=0..n-1} A0(t, i) * A0(t, j)

    RDom t(0, n, "t");
    B(j, i) = cast<num_t>(0);
    B(j, i) += A0(t, i) * A0(t, j);

    // ----------------------
    // Scheduling for init:
    // ----------------------
    // We explicitly compute A0 once at root, then compute B in parallel
    // across rows and vectorize across columns; the reduction over t is
    // also vectorized in the inner j-dimension. This greatly improves
    // cache use and enables SIMD for the Gram-matrix construction.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // A0 is cheap to compute and re-used across the reduction, so
    // materialize it once and make each row parallel.
    A0.compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // Compute B at root; outer loop over rows parallel, inner over
    // columns vectorized.
    B.compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // Vectorize the reduction update across columns as well so that
    // each inner product uses SIMD along j.
    B.update()
        .vectorize(j, vec_width);

    // JIT-compile and realize directly into A.
    B.compile_jit(target);
    B.realize(A);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Matrix A: original C is DATA_TYPE A[N][N];
    // We map to Halide Buffer with shape (cols, rows) = (n, n)
    Halide::Buffer<num_t, 2> A(n, n);

    // Initialize A
    init_array(n, A);

    // -------------------------------------------------------------------------
    // Translate kernel_lu into Halide-assisted code.
    //
    // Original C kernel:
    //
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j < i; j++) {
    //     for (k = 0; k < j; k++) {
    //       A[i][j] -= A[i][k] * A[k][j];
    //     }
    //     A[i][j] /= A[j][j];
    //   }
    //   for (j = i; j < n; j++) {
    //     for (k = 0; k < i; k++) {
    //       A[i][j] -= A[i][k] * A[k][j];
    //     }
    //   }
    // }
    //
    // This algorithm has loop-carried dependencies along i and j, so we cannot
    // express the full kernel as a single pure Halide Func. Instead, we:
    //
    //   - Keep the outer loops over i and j on the host (as required to
    //     preserve the sequential dependencies).
    //   - Use Halide Funcs to compute each inner summation over k:
    //       sum_{k<j} A[i][k] * A[k][j]
    //       sum_{k<i} A[i][k] * A[k][j]
    //
    //   - We use ImageParam + Param<int> for i and j, and realize a scalar
    //     Func for each (i, j) where needed.
    // -------------------------------------------------------------------------

    // ImageParam representing the current matrix A; note that we never write
    // to A inside Halide; we only read from it in the Funcs below. The
    // in-place updates to A are done on the host after each scalar realization.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    A_param.set(A);

    // Scalar parameters for the current (i, j) indices
    Param<int> i_param("i");
    Param<int> j_param("j");

    // We will want to vectorize the inner products over k, so query a
    // reasonable natural vector width for the current host target.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Func to compute A[i][j] - sum_{k=0}^{j-1} A[i][k] * A[k][j]
    // for the case j < i (the "L" part), before dividing by A[j][j].
    Func left_elem("left_elem");
    {
        // A[i][j] in C corresponds to A_param(j_param, i_param)
        left_elem() = A_param(j_param, i_param);

        // Sum over k = 0 .. j-1:
        // A[i][j] -= A[i][k] * A[k][j]
        //
        // A[i][k] -> A_param(k, i_param)
        // A[k][j] -> A_param(j_param, k)
        RDom k_left(0, j_param, "k_left");
        left_elem() -= A_param(k_left, i_param) * A_param(j_param, k_left);

        // Vectorize the reduction over k for better throughput.
        RVar k_left_outer("k_left_outer"), k_left_inner("k_left_inner");
        left_elem.update()
            .split(k_left, k_left_outer, k_left_inner, vec_width)
            .vectorize(k_left_inner);
    }

    // Func to compute A[i][j] - sum_{k=0}^{i-1} A[i][k] * A[k][j]
    // for the case j >= i (the "U" part).
    Func right_elem("right_elem");
    {
        right_elem() = A_param(j_param, i_param);

        // Sum over k = 0 .. i-1:
        // A[i][j] -= A[i][k] * A[k][j]
        RDom k_right(0, i_param, "k_right");
        right_elem() -= A_param(k_right, i_param) * A_param(j_param, k_right);

        // Vectorize the reduction over k.
        RVar k_right_outer("k_right_outer"), k_right_inner("k_right_inner");
        right_elem.update()
            .split(k_right, k_right_outer, k_right_inner, vec_width)
            .vectorize(k_right_inner);
    }

    // JIT-compile the scalar pipelines once; we will reuse them for all (i, j).
    left_elem.compile_jit(target);
    right_elem.compile_jit(target);

    // Reusable 0-D Buffers to avoid reallocating for each scalar realization.
    Halide::Buffer<num_t> tmp_left, tmp_right;

    // -------------------------------------------------------------------------
    // Run kernel and time it
    // -------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        // First inner loop: j = 0 .. i-1 (compute L factors)
        for (int j = 0; j < i; j++) {
            i_param.set(i);
            j_param.set(j);

            // Compute A[i][j] - sum_{k<j} A[i][k] * A[k][j]
            left_elem.realize(tmp_left);
            num_t val = tmp_left();  // zero-dimensional buffer

            // Divide by A[j][j]
            // Mapping: A[j][j] in C -> A(j, j) in Halide
            val /= A(j, j);

            // Store back into A[i][j] (i.e. A(j, i) in Halide)
            A(j, i) = val;
        }

        // Second inner loop: j = i .. n-1 (compute U factors)
        for (int j = i; j < n; j++) {
            i_param.set(i);
            j_param.set(j);

            // Compute A[i][j] - sum_{k<i} A[i][k] * A[k][j]
            right_elem.realize(tmp_right);
            num_t val = tmp_right();  // zero-dimensional buffer

            // Store back into A[i][j] (no division here)
            A(j, i) = val;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -------------------------------------------------------------------------
    // Optional: print the resulting matrix, in the same order as the C version
    // -------------------------------------------------------------------------
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // A[i][j] in C corresponds to A(j, i) in Halide
                std::cerr << A(j, i) << '\n';
            }
        }
    }

    // Print timing (in seconds)
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}