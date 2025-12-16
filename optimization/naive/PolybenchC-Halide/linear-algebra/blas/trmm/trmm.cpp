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
    Expr m_expr = Expr(m);
    Expr n_expr = Expr(n);

    Expr one  = Expr(num_t(1));
    Expr zero = Expr(num_t(0));

    // A(i,j):
    //   For j < i:  ( (i + j) % m ) / m
    //   For j == i: 1.0
    //   For j > i:  0.0 (never used by the kernel, but we define it anyway)
    // Stored as A(j, i).
    init_A(j, i) =
        select(j < i,
               cast<num_t>(((i + j) % m_expr)) / cast<num_t>(m_expr),
               select(j == i,
                      cast<num_t>(one),
                      cast<num_t>(zero)));

    // B(i,j): ( (n + (i - j)) % n ) / n
    // Stored as B(j, i).
    init_B(j, i) =
        cast<num_t>(((n_expr + (i - j)) % n_expr)) / cast<num_t>(n_expr);

    // Simple but effective schedule:
    //  - parallelize over rows (i)
    //  - vectorize across columns (j), using the natural vector width
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    init_A
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    init_B
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // Realize into the provided buffers
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

    // For convenience as Expr
    Expr m_expr = Expr(m);

    // Reduction domain over k in [i+1, m)
    //
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
    // original B (no in-place dependence across rows), so we can express
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
    // We implement "k from i+1..M-1" directly via an RDom whose min and
    // extent depend on the pure variable i. This avoids an inner select()
    // and keeps the inner loop tight:
    //
    //   k runs from (i + 1) for (m - i - 1) iterations.
    //
    RDom k(i + 1, m_expr - (i + 1), "k");

    // Func representing the result B := alpha * A^T * B
    Func trmm("trmm");

    Expr alpha_expr = Expr(alpha);

    trmm(j, i) = cast<num_t>(alpha_expr) *
                 (B_param(j, i) +
                  sum(A_param(i, k) * B_param(j, k)));

    // ------------------------------------------------------------------
    // Scheduling
    // ------------------------------------------------------------------
    //
    // Goals:
    //   - j is the innermost dimension and contiguous in memory,
    //     so vectorize across j.
    //   - i is the outer (row) dimension, independent across rows,
    //     so parallelize across i.
    //
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    // Compute the whole result as a single stage at root.
    trmm
        .compute_root()
        // Parallelize across rows (i). Each (i, j) is independent because
        // we expressed the kernel in closed form over the original B.
        .parallel(i)
        // Vectorize across columns (j) for good SIMD utilization.
        .vectorize(j, vec_width);

    // Compile the kernel to machine code for the chosen target.
    trmm.compile_jit(target);

    // Output buffer for the result (same logical layout as B)
    Buffer<num_t, 2> B_result(n, m);

    // Time the execution (includes first-time JIT compile cost)
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