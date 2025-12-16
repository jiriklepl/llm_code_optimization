#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common Polybench-style definitions (DATA_TYPE, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions for cholesky (DATA_TYPE, N, etc.)
#include "cholesky.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// Initialize matrix A as in the original Polybench C code, then make it
// symmetric positive-definite. We keep the original indexing convention:
//
//   - C code uses A[i][j] (row-major).
//   - Here we use a Halide::Buffer<num_t, 2> A with A(j, i) == A[i][j].
//
// So the first index is the column j, the second index is the row i.
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    Var i("i"), j("j");

    // ------------------------------------------------------------------
    // 1) Build the initial lower-triangular matrix with unit diagonal.
    //
    // Original C:
    //   for (i = 0; i < n; i++) {
    //     for (j = 0; j <= i; j++)
    //       A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
    //     for (j = i+1; j < n; j++)
    //       A[i][j] = 0;
    //     A[i][i] = 1;
    //   }
    //
    // Mapping to Halide Buffer: A(j, i) == A[i][j].
    // ------------------------------------------------------------------
    Func init_tri("init_tri");

    // Compute the base value used in the lower triangle.
    Expr j_mod_n = j % n;
    Expr val = cast<num_t>(-j_mod_n);
    val = val / cast<num_t>(n) + cast<num_t>(1);

    // Lower-triangular (including diagonal) gets 'val', above-diagonal is 0.
    init_tri(j, i) = select(j <= i, val, cast<num_t>(0));

    // Explicitly set the diagonal to 1 (matches the original code).
    init_tri(i, i) = cast<num_t>(1);

    // Simple but effective schedule: vectorize across columns (j) and
    // parallelize across rows (i). This initializes the N x N matrix
    // efficiently.
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    init_tri.compute_root();
    init_tri.reorder(j, i);
    if (vec_width > 1) {
        init_tri.vectorize(j, vec_width);
    }
    init_tri.parallel(i);

    // Realize into the provided buffer A.
    init_tri.realize(A);

    // ------------------------------------------------------------------
    // 2) Make the matrix symmetric positive semi-definite:
    //
    //    B = A * A^T, then A <- B.
    //
    // C code:
    //   for (r = 0; r < n; ++r)
    //     for (s = 0; s < n; ++s) {
    //       B[r][s] = 0;
    //       for (t = 0; t < n; ++t)
    //         B[r][s] += A[r][t] * A[s][t];
    //     }
    //   for (r, s) A[r][s] = B[r][s];
    //
    // Mapping with A(j, i) == A[i][j]:
    //   A[r][t] -> A(t, r)
    //   A[s][t] -> A(t, s)
    //   B[r][s] -> B(s, r) if B(s, r) == B[r][s] in C layout.
    //
    // We perform this as a Halide reduction and then copy back.
    // ------------------------------------------------------------------
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    A_param.set(A);

    Var r("r"), s("s");
    RDom t(0, n, "t");

    Func B_func("B_func");
    B_func(s, r) = cast<num_t>(0);
    // B[r][s] += A[r][t] * A[s][t]  ->  B_func(s, r) += A_param(t, r) * A_param(t, s)
    B_func(s, r) += A_param(t, r) * A_param(t, s);

    // Schedule B_func like a simple GEMM kernel:
    // - compute_root: materialize the whole result
    // - reorder so that s (columns) is innermost
    // - vectorize across s, parallelize across r
    B_func.compute_root();
    B_func.reorder(s, r);
    if (vec_width > 1) {
        B_func.vectorize(s, vec_width);
    }
    B_func.parallel(r);

    // For the update (the reduction over t), use the same parallel/vec pattern.
    B_func.update()
        .reorder(s, t, r);
    if (vec_width > 1) {
        B_func.update().vectorize(s, vec_width);
    }
    B_func.update().parallel(r);

    Halide::Buffer<num_t, 2> B(n, n);
    B_func.realize(B);

    // Copy B back into A (same layout A(j, i) == B[j][i]).
    for (int rr = 0; rr < n; ++rr) {
        for (int ss = 0; ss < n; ++ss) {
            A(ss, rr) = B(ss, rr);
        }
    }
}

int main(int argc, char *argv[]) {
    // Problem size.
    int n = N;

    // Matrix A, using Halide::Buffer.
    // Convention: A(j, i) == A[i][j] from the original C code.
    Halide::Buffer<num_t, 2> A(n, n);

    // Initialize A as in the original Polybench code, then form A = A * A^T.
    init_array(n, A);

    // ----------------------------------------------------------------------
    // Build a Halide pipeline to compute inner products (dot products) of
    // rows of A. This pipeline is used inside the Cholesky factorization
    // loops to accelerate the O(N^3) work in the k-dimension via SIMD.
    //
    // We still respect the strict loop-carried dependencies in Cholesky:
    //   - Rows i must be processed in increasing order.
    //   - For a given i, columns j < i must be processed in increasing j.
    //
    // Therefore, we keep the outer (i, j) loops in C++, but we use Halide
    // to implement the inner dot-product:
    //
    //   sum_{k=0}^{L-1} A[i][k] * A[j][k]
    //
    // where L = j for off-diagonals and L = i for the diagonal update.
    //
    // In Buffer layout this is:
    //   sum_{k=0}^{L-1} A(k, i) * A(k, j)
    // (iterate over columns k for fixed rows i and j).
    // ----------------------------------------------------------------------
    ImageParam A_param_factor(type_of<num_t>(), 2, "A_param_factor");
    A_param_factor.set(A);  // bind to the same underlying storage as A

    Param<int> row_i("row_i");
    Param<int> row_j("row_j");
    Param<int> limit("limit");

    Var idx("idx");
    RDom k_r(0, limit, "k_r");  // reduction domain [0, limit)

    Func cholesky_dot("cholesky_dot");

    // cholesky_dot is a 1D Func, but we only ever use index 0.
    cholesky_dot(idx) = cast<num_t>(0);
    // Accumulate the dot product into cholesky_dot(0).
    cholesky_dot(0) += A_param_factor(k_r, row_i) * A_param_factor(k_r, row_j);

    // Schedule: compute_root so it is fully materialized once per call.
    // Vectorize over the reduction variable k_r to get SIMD inner products.
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    cholesky_dot.compute_root();
    if (vec_width > 1) {
        cholesky_dot.update().vectorize(k_r, vec_width);
    }

    // JIT-compile the dot-product pipeline.
    // Set default parameter values to avoid uninitialized usage.
    row_i.set(0);
    row_j.set(0);
    limit.set(0);
    cholesky_dot.compile_jit(target);

    // Preallocate a small buffer to hold the scalar dot-product result.
    // Only element cholesky_dot(0) is actually written.
    Halide::Buffer<num_t> dot_buf(1);

    // ----------------------------------------------------------------------
    // Time the Cholesky factorization kernel (the part Polybench measures).
    // ----------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // Original C kernel for reference (in terms of A[i][j]):
    //
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j < i; j++) {
    //     for (k = 0; k < j; k++) {
    //       A[i][j] -= A[i][k] * A[j][k];
    //     }
    //     A[i][j] /= A[j][j];
    //   }
    //   for (k = 0; k < i; k++) {
    //     A[i][i] -= A[i][k] * A[i][k];
    //   }
    //   A[i][i] = sqrt(A[i][i]);
    // }
    //
    // With our Buffer mapping A(j, i) == A[i][j], and using the Halide
    // dot-product pipeline for the inner sums, this becomes:
    //
    for (int i = 0; i < n; i++) {
        // Off-diagonal elements in row i: j = 0 .. i-1.
        for (int j = 0; j < i; j++) {
            // Compute sum_{k=0}^{j-1} A(i,k) * A(j,k) via Halide:
            // A(i,k) -> A(k, i), A(j,k) -> A(k, j).
            row_i.set(i);
            row_j.set(j);
            limit.set(j);  // sum over k in [0, j)

            cholesky_dot.realize(dot_buf);
            num_t dot_ij = dot_buf(0);

            // A[i][j] -= dot_ij; A[i][j] /= A[j][j];
            // -> A(j, i) -= dot_ij; A(j, i) /= A(j, j);
            A(j, i) -= dot_ij;
            A(j, i) /= A(j, j);
        }

        // Diagonal element: A[i][i] -= sum_{k=0}^{i-1} A[i][k]^2;
        // then A[i][i] = sqrt(A[i][i]).
        if (i > 0) {
            row_i.set(i);
            row_j.set(i);
            limit.set(i);  // sum over k in [0, i)

            cholesky_dot.realize(dot_buf);
            num_t dot_ii = dot_buf(0);

            // A[i][i] -= dot_ii; -> A(i, i) -= dot_ii;
            A(i, i) -= dot_ii;
        }

        A(i, i) = static_cast<num_t>(
            std::sqrt(static_cast<double>(A(i, i)))
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the resulting lower-triangular matrix A,
    // similar to Polybench's print_array (only j <= i).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                std::cerr << A(j, i) << '\n';
            }
        }
    }

    // Print elapsed time in seconds.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}