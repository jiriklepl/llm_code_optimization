#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common PolyBench-style definitions (DATA_TYPE, NI, NJ, NK, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (for GEMM)
#include "gemm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize the arrays A, B, C exactly as in the original C code,
 * but using Halide Funcs and Buffers instead of raw C arrays.
 *
 * Original C loops:
 *   *alpha = 1.5;
 *   *beta  = 1.2;
 *   for (i = 0; i < ni; i++)
 *     for (j = 0; j < nj; j++)
 *       C[i][j] = ((i*j+1) % ni) / ni;
 *   for (i = 0; i < ni; i++)
 *     for (j = 0; j < nk; j++)
 *       A[i][j] = (i*(j+1) % nk) / nk;
 *   for (i = 0; i < nk; i++)
 *     for (j = 0; j < nj; j++)
 *       B[i][j] = (i*(j+2) % nj) / nj;
 *
 * Mapping to Halide:
 *   - C[i][j]  (C is NI x NJ)  -> Buffer C(j, i)  of shape (nj, ni)
 *   - A[i][j]  (A is NI x NK)  -> Buffer A(k, i)  of shape (nk, ni)
 *   - B[i][j]  (B is NK x NJ)  -> Buffer B(j, k)  of shape (nj, nk)
 */
static void init_array(int ni, int nj, int nk,
                       num_t *alpha,
                       num_t *beta,
                       Halide::Buffer<num_t, 2> C,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j"), k("k");
    Func init_A("init_A"), init_B("init_B"), init_C("init_C");

    // C[i][j] = ((i*j+1) % ni) / ni;
    // Buffer layout: C(j, i)
    init_C(j, i) = cast<num_t>((i * j + 1) % ni) / cast<int>(ni);

    // A[i][j] = (i*(j+1) % nk) / nk;
    // Buffer layout: A(k, i) where k is the "j" from the C code
    init_A(k, i) = cast<num_t>(i * (k + 1) % nk) / cast<int>(nk);

    // B[i][j] = (i*(j+2) % nj) / nj;
    // Buffer layout: B(j, k) where i in C code is our k, and j is j
    init_B(j, k) = cast<num_t>(k * (j + 2) % nj) / cast<int>(nj);

    // Realize initialization into the provided buffers
    init_C.realize(C);
    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem sizes (as in PolyBench)
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers:
    //   C: NI x NJ => (width = nj, height = ni)
    //   A: NI x NK => (width = nk, height = ni)
    //   B: NK x NJ => (width = nj, height = nk)
    Halide::Buffer<num_t, 2> C(nj, ni);
    Halide::Buffer<num_t, 2> B(nj, nk);
    Halide::Buffer<num_t, 2> A(nk, ni);

    // ImageParams to represent the input/output images for the Halide pipeline.
    // They share the same logical layout as the Buffers above.
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");

    // Initialize data as in the original C code.
    init_array(ni, nj, nk, &alpha, &beta, C, A, B);

    // Define Vars
    Var i("i"), j("j");

    // Reduction domain over k = 0..nk-1 (the shared dimension of A and B)
    RDom k(0, A.dim(0).extent(), "k");

    // Main GEMM computation:
    //
    // Original C kernel:
    //
    // for (i = 0; i < ni; i++) {
    //   for (j = 0; j < nj; j++)
    //       C[i][j] *= beta;
    //   for (k = 0; k < nk; k++) {
    //      for (j = 0; j < nj; j++)
    //         C[i][j] += alpha * A[i][k] * B[k][j];
    //   }
    // }
    //
    // This is equivalent to:
    //   C[i][j] = beta * C[i][j] +
    //             sum_{k=0..nk-1}(alpha * A[i][k] * B[k][j]);
    //
    // Mapping to Halide coordinates:
    //   C(i,j) => C_param(j, i)
    //   A(i,k) => A_param(k, i)
    //   B(k,j) => B_param(j, k)
    //
    // We'll implement C := beta*C + alpha*A*B using a reduction.
    Func gemm("gemm");

    // Pure definition: initialize each output element with beta * C
    gemm(j, i) = Expr(beta) * C_param(j, i);

    // Update definition: accumulate alpha * A * B over k
    gemm(j, i) += Expr(alpha) * A_param(k, i) * B_param(j, k);

    // Schedule:
    // Make j the innermost loop, i the outermost (row-major in terms of C[i][j]).
    gemm.reorder(j, i);
    gemm.update().reorder(j, k, i);

    // Bind the ImageParams to the concrete Buffers
    C_param.set(C);
    A_param.set(A);
    B_param.set(B);

    // Compile the kernel to machine code (JIT)
    gemm.compile_jit();

    // Measure execution time of the GEMM kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write the result into C
    gemm.realize(C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally dump the resulting C, similar to PolyBench's print_array.
    // Here we just print all values to stderr when invoked with any argv[0] (always true),
    // but keep the check to mirror the example structure.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < ni; ii++) {
            for (int jj = 0; jj < nj; jj++) {
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    // Print timing in seconds to stdout
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}