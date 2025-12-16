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

// Initialize matrix A as in the original Polybench C code.
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    // A is laid out as A(j, i) == A[i][j] from the C code.
    Var i("i"), j("j");

    // First construct the lower-triangular matrix with diagonal 1,
    // as in the original init_array.
    //
    // C code:
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j <= i; j++)
    //     A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
    //   for (j = i+1; j < n; j++)
    //     A[i][j] = 0;
    //   A[i][i] = 1;
    // }
    //
    // Mapping to Halide: A(j, i) corresponds to A[i][j].
    Func init_tri("init_tri");

    Expr val = cast<num_t>(-(j % n));
    val = val / cast<num_t>(n) + cast<num_t>(1);

    // Lower-triangular (including diagonal) gets 'val', above-diagonal is 0.
    init_tri(j, i) = select(j <= i, val, cast<num_t>(0));

    // Explicitly set the diagonal to 1.
    init_tri(i, i) = cast<num_t>(1);

    // Realize into the provided buffer A.
    init_tri.realize(A);

    // Now make the matrix positive semi-definite:
    //
    // C code:
    // int r, s, t;
    // B[r][s] = 0;
    // for (t = 0; t < n; ++t)
    //   for (r = 0; r < n; ++r)
    //     for (s = 0; s < n; ++s)
    //       B[r][s] += A[r][t] * A[s][t];
    // A[r][s] = B[r][s];
    //
    // This is B = A * A^T where A is n x n lower-triangular.

    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    A_param.set(A);

    Var r("r"), s("s");
    RDom t(0, n, "t");

    Func B_func("B_func");
    B_func(s, r) = cast<num_t>(0);
    // A[r][t] -> A_param(t, r);  A[s][t] -> A_param(t, s)
    B_func(s, r) += A_param(t, r) * A_param(t, s);

    Halide::Buffer<num_t, 2> B(n, n);
    B_func.realize(B);

    // Copy B back into A.
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
    // We follow the convention: A(j, i) == A[i][j] from the original C.
    Halide::Buffer<num_t, 2> A(n, n);

    // Initialize A as in the original Polybench code.
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