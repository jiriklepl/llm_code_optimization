#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, N, M, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "syr2k.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize arrays A, B, C and scalars alpha, beta.
 * This mirrors the original C init_array() function.
 */
static void init_array(int n, int m,
                       num_t *alpha,
                       num_t *beta,
                       Halide::Buffer<num_t, 2> C,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    // We will initialize using small Halide pipelines.
    Var i("i"), j("j");

    // Original C code:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < m; j++) {
    //     A[i][j] = ((i*j+1)%n) / n;
    //     B[i][j] = ((i*j+2)%m) / m;
    //   }
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < n; j++)
    //     C[i][j] = ((i*j+3)%n) / m;
    //
    // Mapping C[i][j] -> Buffer(j, i),
    // A[i][j] (n x m) -> Buffer(j, i) with shape (m, n),
    // B[i][j] similarly.

    Func init_A("init_A"), init_B("init_B"), init_C("init_C");

    // A has shape (m, n), accessed as A(j, i) == A[i][j]
    init_A(j, i) = cast<num_t>(((i * j + 1) % n)) / cast<int>(n);

    // B has shape (m, n), accessed as B(j, i) == B[i][j]
    init_B(j, i) = cast<num_t>(((i * j + 2) % m)) / cast<int>(m);

    // C has shape (n, n), accessed as C(j, i) == C[i][j]
    init_C(j, i) = cast<num_t>(((i * j + 3) % n)) / cast<int>(m);

    init_A.realize(A);
    init_B.realize(B);
    init_C.realize(C);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;
    int m = M;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers
    // Original arrays:
    //   C[n][n], A[n][m], B[n][m]
    // We map C[i][j] -> C(j, i) with shape (n, n)
    // A[i][j] (n x m) -> A(j, i) with shape (m, n)
    // B[i][j] (n x m) -> B(j, i) with shape (m, n)
    Buffer<num_t, 2> C(n, n);
    Buffer<num_t, 2> A(m, n);
    Buffer<num_t, 2> B(m, n);

    // ImageParams to feed buffers into Halide pipeline
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");

    // Initialize data
    init_array(n, m, &alpha, &beta, C, A, B);

    // Bind buffers to ImageParams
    C_param.set(C);
    A_param.set(A);
    B_param.set(B);

    // Vars and reduction domain
    Var i("i"), j("j");
    RDom k(0, A.dim(0).extent(), "k"); // A.dim(0) == m

    // Halide implementation of kernel_syr2k:
    //
    // Original C kernel:
    //
    // for (i = 0; i < _PB_N; i++) {
    //   for (j = 0; j <= i; j++)
    //     C[i][j] *= beta;
    //   for (k = 0; k < _PB_M; k++)
    //     for (j = 0; j <= i; j++) {
    //       C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
    //     }
    // }
    //
    // We compute for all (i,j) in [0..n-1]x[0..n-1], but apply the update
    // only when j <= i. For j > i, C[i][j] remains as initialized.

    Func syr2k("syr2k");

    // Mapping:
    //   C[i][j]   -> C_param(j, i)
    //   A[j][k]   -> A_param(k, j)
    //   B[i][k]   -> B_param(k, i)
    //   B[j][k]   -> B_param(k, j)
    //   A[i][k]   -> A_param(k, i)

    Expr alpha_e = Expr(alpha);
    Expr beta_e  = Expr(beta);

    // Initial multiplication by beta for j <= i; otherwise keep original C
    syr2k(j, i) = select(j <= i,
                         beta_e * C_param(j, i),
                         C_param(j, i));

    // Rank-2k update, only for j <= i
    syr2k(j, i) += select(
        j <= i,
        alpha_e * (A_param(k, j) * B_param(k, i) +
                   B_param(k, j) * A_param(k, i)),
        cast<num_t>(0));

    // Schedule: keep it simple but roughly reflect original loop ordering.
    // Pure definition iterates over i (rows) outer, j (cols) inner.
    syr2k.reorder(j, i);

    // Update definition: iterate over i outer, then k, then j.
    syr2k.update().reorder(j, k, i);

    // JIT-compile the kernel
    syr2k.compile_jit();

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write result back into C
    syr2k.realize(C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result (mainly to prevent dead-code elimination
    // and to allow correctness checking if desired).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // Recall: C(ii,jj) in C maps to C(jj,ii) in Buffer
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}