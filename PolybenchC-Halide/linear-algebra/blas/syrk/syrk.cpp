#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "syrk.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Array initialization, translated from the original C version.
 */
static
void init_array(int n, int m,
                num_t *alpha,
                num_t *beta,
                Halide::Buffer<num_t, 2> C,
                Halide::Buffer<num_t, 2> A) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j");
    Func init_A("init_A"), init_C("init_C");

    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < m; j++)
    //     A[i][j] = ((i*j+1)%n) / n;
    //
    // C array A[i][j] is mapped to Buffer A(j, i) with shape (m, n).
    init_A(j, i) = cast<num_t>((i * j + 1) % n) / cast<int>(n);

    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < n; j++)
    //     C[i][j] = ((i*j+2)%m) / m;
    //
    // C array C[i][j] is mapped to Buffer C(j, i) with shape (n, n).
    init_C(j, i) = cast<num_t>((i * j + 2) % m) / cast<int>(m);

    init_A.realize(A);
    init_C.realize(C);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;
    int m = M;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers:
    // C is N x N in C as C[i][j], map to Buffer C(j, i) => shape (n, n)
    // A is N x M in C as A[i][j], map to Buffer A(j, i) => shape (m, n)
    Halide::Buffer<num_t, 2> C(n, n);
    Halide::Buffer<num_t, 2> A(m, n);

    // ImageParams corresponding to C and A
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");

    // Initialize data using Halide
    init_array(n, m, &alpha, &beta, C, A);

    // Halide Vars and RDom
    Var i("i"), j("j");
    RDom k(0, A.dim(0).extent(), "k"); // k over M (first dimension of A)

    // Build the SYRK kernel in Halide
    // BLAS SYRK:
    // C := alpha * A * A^T + beta * C
    //
    // Original C kernel:
    // for (i = 0; i < _PB_N; i++) {
    //   for (j = 0; j <= i; j++)
    //     C[i][j] *= beta;
    //   for (k = 0; k < _PB_M; k++) {
    //     for (j = 0; j <= i; j++)
    //       C[i][j] += alpha * A[i][k] * A[j][k];
    //   }
    // }
    //
    // We express this as:
    // C_new[i][j] = C[i][j]                       if j > i
    //             = beta * C[i][j] + alpha * sum_k A[i][k] * A[j][k]  if j <= i
    //
    // Remember mapping:
    //   C[i][j]   -> C_param(j, i)
    //   A[i][k]   -> A_param(k, i)
    //   A[j][k]   -> A_param(k, j)
    //
    // We'll compute a full N x N Func, and use select(j <= i, ..., orig)
    // to restrict updates to the lower triangle.

    Func syrk("syrk");

    Expr alpha_expr = Expr(alpha);
    Expr beta_expr  = Expr(beta);

    Expr orig = C_param(j, i);

    // Pure definition: scale lower triangle by beta, keep upper triangle unchanged.
    syrk(j, i) = select(j <= i,
                        cast<num_t>(beta_expr) * orig,
                        orig);

    // Reduction over k: add alpha * A[i][k] * A[j][k] for lower triangle.
    syrk(j, i) += select(j <= i,
                         cast<num_t>(alpha_expr) *
                             A_param(k, i) * A_param(k, j),
                         cast<num_t>(0));

    // Schedule: match the original traversal order as much as practical.
    // Original C loops are i outer, j inner (up to i), then k inside.
    // Our storage is (j, i), so we reorder to make j the innermost loop.
    syrk.reorder(j, i);
    syrk.update().reorder(j, k, i);

    // Bind ImageParams
    C_param.set(C);
    A_param.set(A);

    // Compile the kernel to machine code
    syrk.compile_jit();

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write result into C (in-place semantics w.r.t the original code)
    syrk.realize(C);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<long double>(end - start);

    // Print results (similar spirit to the original print_array), guarded
    // the same way as in the GEMM example.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // C array C[ii][jj] -> Buffer C(jj, ii)
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}