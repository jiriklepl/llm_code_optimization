#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common PolyBench definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (M, N, etc.)
#include "trmm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize A and B with the same semantics as the original C init_array().
 *
 * C version:
 *
 *   *alpha = 1.5;
 *   for (i = 0; i < m; i++) {
 *     for (j = 0; j < i; j++) {
 *       A[i][j] = (DATA_TYPE)((i+j) % m)/m;
 *     }
 *     A[i][i] = 1.0;
 *     for (j = 0; j < n; j++) {
 *       B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
 *     }
 *   }
 *
 * Mapping to Halide Buffers:
 *   - A is declared in C as A[m][m] and accessed as A[i][j].
 *     We store it as Buffer A(width=m, height=m) and access as A(j, i).
 *   - B is declared in C as B[m][n] and accessed as B[i][j].
 *     We store it as Buffer B(width=n, height=m) and access as B(j, i).
 */
static void init_array(int m, int n,
                       num_t *alpha,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B) {
    *alpha = static_cast<num_t>(1.5);

    Var i("i"), j("j");
    Func init_A("init_A"), init_B("init_B");

    // Make Expr versions of m and n for use inside Halide expressions.
    Expr m_expr = m;
    Expr n_expr = n;

    // A(i,j):
    //   For j < i:  ( (i + j) % m ) / m
    //   For j == i: 1.0
    //   For j > i:  0.0 (never used by the kernel, but we define it anyway)
    init_A(j, i) =
        select(j < i,
               cast<num_t>(((i + j) % m_expr)) / cast<num_t>(m_expr),
               select(j == i,
                      static_cast<num_t>(1.0),
                      static_cast<num_t>(0.0)));

    // B(i,j): ( (n + (i - j)) % n ) / n
    init_B(j, i) =
        cast<num_t>(((n_expr + (i - j)) % n_expr)) / cast<num_t>(n_expr);

    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem sizes (compile-time constants from PolyBench headers)
    int m = M;
    int n = N;

    // Scalars and arrays
    num_t alpha;
    // Buffer layout: width = number of columns, height = number of rows
    // A: M x M  -> Buffer(m, m) accessed as A(j, i) == A[i][j] in C
    // B: M x N  -> Buffer(n, m) accessed as B(j, i) == B[i][j] in C
    Buffer<num_t, 2> A(m, m);
    Buffer<num_t, 2> B(n, m);

    // Initialize data
    init_array(m, n, &alpha, A, B);

    // Halide ImageParams corresponding to A and B (read-only views)
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");
    A_param.set(A);
    B_param.set(B);

    // Vars
    Var i("i"), j("j");

    // Reduction domain over k in [0, m)
    RDom k(0, m, "k");

    // Func representing the result B := alpha * A^T * B
    // Original kernel:
    //
    // for (i = 0; i < M; i++)
    //   for (j = 0; j < N; j++) {
    //     for (k = i+1; k < M; k++)
    //       B[i][j] += A[k][i] * B[k][j];
    //     B[i][j] = alpha * B[i][j];
    //   }
    //
    // Careful dependency analysis shows that all reads of B[k][j] use the
    // *original* B (no in-place dependence across rows), so we can express
    // the final value in closed form as:
    //
    //   B_out[i][j] = alpha * ( B_in[i][j] +
    //                           sum_{k = i+1..M-1} A[k][i] * B_in[k][j] )
    //
    // Mapping indices to Halide:
    //   A[k][i]  -> A_param(i, k)
    //   B[k][j]  -> B_param(j, k)
    //   B[i][j]  -> B_param(j, i)
    //
    // We implement "k from i+1..M-1" via a select inside a reduction over k in [0..M-1].
    Func trmm("trmm");

    Expr alpha_expr = Expr(alpha);
    Expr zero = cast<num_t>(0);

    trmm(j, i) = alpha_expr *
                 (B_param(j, i) +
                  sum(select(k > i,
                             A_param(i, k) * B_param(j, k),
                             zero)));

    // Optional schedule to make the traversal order explicit:
    // i (rows) outer, j (cols) inner, matching the original C loops.
    trmm.reorder(j, i);

    // Compile the kernel to machine code
    trmm.compile_jit();

    // Output buffer for the result (same logical layout as B)
    Buffer<num_t, 2> B_result(n, m);

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();

    trmm.realize(B_result);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result to avoid dead-code elimination
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < m; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // Remember: B_result(jj, ii) == B[ii][jj] in the original C code
                std::cerr << B_result(jj, ii) << '\n';
            }
        }
    }

    // Print elapsed time in seconds
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}