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

    // Constants as Exprs for use inside Halide expressions
    Expr ni_e = Expr(ni);
    Expr nj_e = Expr(nj);
    Expr nk_e = Expr(nk);

    // C[i][j] = ((i*j+1) % ni) / ni;
    // Buffer layout: C(j, i)
    init_C(j, i) = cast<num_t>(((i * j + 1) % ni)) / ni_e;

    // A[i][j] = (i*(j+1) % nk) / nk;
    // Buffer layout: A(k, i) where k is the "j" from the C code
    init_A(k, i) = cast<num_t>((i * (k + 1) % nk)) / nk_e;

    // B[i][j] = (i*(j+2) % nj) / nj;
    // Buffer layout: B(j, k) where i in C code is our k, and j is j
    init_B(j, k) = cast<num_t>((k * (j + 2) % nj)) / nj_e;

    // ---- Initialization schedules ----------------------------------------
    //
    // These run once, but we still give them reasonably efficient
    // schedules: parallel over the outer (row) dimension, and
    // vectorize across the contiguous inner dimension.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    Var jo("jo"), ji("ji");
    Var ko("ko"), ki("ki");

    // C: shape (nj, ni) => j is innermost, i is outermost
    init_C
        .compute_root()
        .split(j, jo, ji, vec_width)
        .reorder(ji, jo, i)   // innermost: ji (vector), then jo, then i
        .vectorize(ji)
        .parallel(i);

    // A: shape (nk, ni) => k is innermost, i is outermost
    init_A
        .compute_root()
        .split(k, ko, ki, vec_width)
        .reorder(ki, ko, i)
        .vectorize(ki)
        .parallel(i);

    // B: shape (nj, nk) => j is innermost, k is outermost
    init_B
        .compute_root()
        .split(j, jo, ji, vec_width)
        .reorder(ji, jo, k)
        .vectorize(ji)
        .parallel(k);

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
    Var i("i"), j("j"); // i: row index [0..ni-1], j: column index [0..nj-1]

    // Reduction domain over k = 0..nk-1 (the shared dimension of A and B)
    RDom k(0, nk, "k");

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
    //   C[i][j]  -> C_param(j, i)
    //   A[i][k]  -> A_param(k, i)
    //   B[k][j]  -> B_param(j, k)
    //
    // We'll implement C := beta*C + alpha*A*B using a reduction.
    Func gemm("gemm");

    // Pure definition: initialize each output element with beta * C
    //
    // Note: we read from C_param and write into C, which alias the same
    // underlying memory. This is safe because each output element only
    // depends on the original C[i][j] at the same coordinates, and we
    // read that value before overwriting it.
    gemm(j, i) = Expr(beta) * C_param(j, i);

    // Update definition: accumulate alpha * A * B over k
    gemm(j, i) += Expr(alpha) * A_param(k, i) * B_param(j, k);

    // ----------------------- Schedule for GEMM -----------------------------
    //
    // Layout recap:
    //   - C_param(j, i) / gemm(j, i)   : j is the innermost dimension
    //   - A_param(k, i)               : k innermost, i outermost
    //   - B_param(j, k)               : j innermost, k outermost
    //
    // We want:
    //   - j as the innermost loop to get unit-stride access in C and B
    //   - parallelism over i (rows) to exploit multiple cores
    //   - vectorization across j to exploit SIMD
    //
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    Var jo("jo"), ji("ji");

    // Pure step:
    //
    // After split:
    //   j = jo * vec_width + ji
    //
    // Loop order (from outermost to innermost) will be:
    //   for i (parallel)
    //     for jo
    //       for ji (vectorized)
    //
    gemm
        .split(j, jo, ji, vec_width)
        .reorder(ji, jo, i)  // innermost -> outermost
        .vectorize(ji)
        .parallel(i);

    // Update step (reduction over k):
    //
    // Desired loop order:
    //   for i (parallel)
    //     for jo
    //       for ji (vectorized)
    //         for k (reduction, innermost)
    //
    // Reordering from innermost outward as: k, ji, jo, i.
    gemm.update()
        .split(j, jo, ji, vec_width)
        .reorder(k, ji, jo, i)
        .vectorize(ji)
        .parallel(i);

    // Bind the ImageParams to the concrete Buffers
    C_param.set(C);
    A_param.set(A);
    B_param.set(B);

    // Compile the kernel to machine code (JIT) once, outside the timed region
    gemm.compile_jit(target);

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