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

// Initialize matrix A as in the original Polybench C code, but do it
// as an optimized Halide pipeline.
//
// Semantics (original C):
//   1) Build a lower-triangular matrix L with unit diagonal:
//        for (i = 0; i < n; i++) {
//          for (j = 0; j <= i; j++)
//            A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
//          for (j = i+1; j < n; j++)
//            A[i][j] = 0;
//          A[i][i] = 1;
//        }
//   2) Form B = L * L^T, then A = B (symmetric positive semi-definite).
//
// We keep the mapping A(j, i) == A[i][j] from the C code (i = row, j = col).
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    // --------------------------
    // 1. Define the Halide Vars.
    // --------------------------
    Var i("i"), j("j");       // For the lower-triangular generator L = tri.
    Var r("r"), s("s");       // For the SPD matrix B(r,s).
    RDom k(0, n, "k");        // Reduction domain over the inner dimension.

    // ----------------------------------------------------------
    // 2. Construct the lower-triangular matrix L as a Halide Func
    // ----------------------------------------------------------
    //
    // tri(j, i) corresponds to A[i][j] in the reference C code.
    //
    // Reference initialization when j <= i:
    //   val = (DATA_TYPE)(-j % n) / n + 1;
    //   A[i][j] = val;
    // For j > i: A[i][j] = 0;
    // Then explicitly force A[i][i] = 1.
    //
    // Note: in our domain j is always in [0, n-1], so (j % n) == j.
    // We can safely drop the modulo without changing semantics.
    Func tri("tri");

    Expr n_f   = cast<num_t>(n);
    Expr one   = cast<num_t>(1);
    Expr j_f   = cast<num_t>(j);
    Expr val   = (-j_f) / n_f + one;  // == (DATA_TYPE)(-j) / n + 1

    // Default: lower triangle (strict) gets 'val'; everything else is zero.
    tri(j, i) = select(j < i, val, cast<num_t>(0));
    // Override the diagonal to be exactly 1.
    tri(i, i) = cast<num_t>(1);

    // ----------------------------------------------------------
    // 3. Build B = L * L^T via a matrix-multiply-style reduction
    // ----------------------------------------------------------
    //
    // Original C:
    //   B[r][s] = 0;
    //   for (t = 0; t < n; ++t)
    //     B[r][s] += A[r][t] * A[s][t];
    //
    // Mapping with A(j, i) == A[i][j]:
    //   A[r][t] -> tri(t, r)
    //   A[s][t] -> tri(t, s)
    //   B[r][s] -> B_func(s, r) (col, row)
    Func B_func("B_func");
    B_func(s, r) = cast<num_t>(0);
    B_func(s, r) += tri(k, r) * tri(k, s);

    // ----------------------------------------------------------
    // 4. Schedule for better locality and parallelism
    // ----------------------------------------------------------
    //
    // Strategy:
    //   - Compute tri once at root so it can be reused efficiently.
    //   - Parallelize across rows (i, r) and vectorize across columns (j, s),
    //     which are contiguous in memory (dim 0 in Halide).
    //
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // tri(j, i): j is innermost/contiguous; i is outer.
    tri.compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // B_func(s, r): s is innermost/contiguous; r is outer.
    B_func.compute_root()
        .parallel(r)           // parallelize across rows of B
        .vectorize(s, vec_width);

    // Make sure the update (reduction) stage also uses the same
    // parallel/vectorized structure for good performance.
    B_func.update()
        .parallel(r)
        .vectorize(s, vec_width);

    // ----------------------------------------------------------
    // 5. Realize directly into A
    // ----------------------------------------------------------
    //
    // B_func(s, r) is B[r][s] in C indexing, so A(s, r) == A[i][j]
    // will contain the SPD matrix as required:
    //
    //   A[i][j] = B[i][j] = sum_t L[i][t] * L[j][t]
    //
    // No extra copying is needed.
    B_func.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size.
    int n = N;

    // Matrix A, using Halide::Buffer.
    // We follow the convention: A(j, i) == A[i][j] from the original C.
    Halide::Buffer<num_t, 2> A(n, n);

    // Initialize A as in the original Polybench code (now with
    // an optimized Halide pipeline).
    init_array(n, A);

    // Time the Cholesky factorization kernel.
    auto start = std::chrono::high_resolution_clock::now();

    // Main computational kernel translated directly, operating on the Buffer.
    //
    // Original C:
    //
    // for (i = 0; i < _PB_N; i++) {
    //   // j < i
    //   for (j = 0; j < i; j++) {
    //     for (k = 0; k < j; k++) {
    //       A[i][j] -= A[i][k] * A[j][k];
    //     }
    //     A[i][j] /= A[j][j];
    //   }
    //   // i == j case
    //   for (k = 0; k < i; k++) {
    //     A[i][i] -= A[i][k] * A[i][k];
    //   }
    //   A[i][i] = SQRT_FUN(A[i][i]);
    // }
    //
    // With mapping A(j, i) == A[i][j]:
    //   A[i][j]   -> A(j, i)
    //   A[i][k]   -> A(k, i)
    //   A[j][k]   -> A(k, j)
    //   A[j][j]   -> A(j, j)
    //   A[i][i]   -> A(i, i)
    //
    // The dependency pattern here (Cholesky factorization) is highly
    // sequential along i and j, so we keep this part in straightforward
    // C++ loops for correctness; most of the heavy O(n^3) work in
    // initialization (matrix multiply to make A SPD) has been moved
    // into a parallel, vectorized Halide pipeline above.
    for (int i = 0; i < n; i++) {
        // Off-diagonal elements in row i.
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A(j, i) -= A(k, i) * A(k, j);
            }
            A(j, i) /= A(j, j);
        }

        // Diagonal element.
        for (int k = 0; k < i; k++) {
            A(i, i) -= A(k, i) * A(k, i);
        }
        A(i, i) = static_cast<num_t>(std::sqrt(static_cast<double>(A(i, i))));
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