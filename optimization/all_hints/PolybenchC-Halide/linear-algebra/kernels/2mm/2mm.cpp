#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common Polybench definitions (NI, NJ, NK, NL, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions for 2mm
#include "2mm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// -----------------------------------------------------------------------------
// Tunable scheduling constants
// -----------------------------------------------------------------------------

// Vector width for x‑dimension vectorization. 8 works well for AVX2-sized
// vectors on modern x86 for float; for double it will use two AVX2 vectors.
// Adjust if desired for a particular machine.
constexpr int VEC_WIDTH = 8;

// -----------------------------------------------------------------------------
// Initialize arrays A, B, C, D in the same way as the original C version.
//
// C layout:
//   A[ni][nk] -> Buffer A(k, i)  with shape (nk, ni)
//   B[nk][nj] -> Buffer B(j, k)  with shape (nj, nk)
//   C[nj][nl] -> Buffer C(l, j)  with shape (nl, nj)
//   D[ni][nl] -> Buffer D(l, i)  with shape (nl, ni)
// -----------------------------------------------------------------------------
static void init_array(int ni, int nj, int nk, int nl,
                       num_t *alpha,
                       num_t *beta,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B,
                       Halide::Buffer<num_t, 2> C,
                       Halide::Buffer<num_t, 2> D) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j"), k("k"), l("l");
    Func init_A("init_A"), init_B("init_B"), init_C("init_C"), init_D("init_D");

    // A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
    // A is stored as A(k, i) with shape (nk, ni)
    init_A(k, i) = cast<num_t>(((i * k + 1) % ni)) / cast<int>(ni);

    // B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
    // B is stored as B(j, k) with shape (nj, nk)
    init_B(j, k) = cast<num_t>(((k * (j + 1)) % nj)) / cast<int>(nj);

    // C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
    // C is stored as C(l, j) with shape (nl, nj)
    init_C(l, j) = cast<num_t>(((j * (l + 3) + 1) % nl)) / cast<int>(nl);

    // D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
    // D is stored as D(l, i) with shape (nl, ni)
    init_D(l, i) = cast<num_t>(((i * (l + 2)) % nk)) / cast<int>(nk);

    // Simple yet cache‑friendly schedules for initialization.
    // These are not part of the timed kernel, but we keep them efficient.

    // A: rows over i (parallel), contiguous over k (vectorized)
    init_A
        .compute_root()
        .reorder(k, i)
        .parallel(i)
        .vectorize(k, VEC_WIDTH);

    // B: rows over k (parallel), contiguous over j (vectorized)
    init_B
        .compute_root()
        .reorder(j, k)
        .parallel(k)
        .vectorize(j, VEC_WIDTH);

    // C: rows over j (parallel), contiguous over l (vectorized)
    init_C
        .compute_root()
        .reorder(l, j)
        .parallel(j)
        .vectorize(l, VEC_WIDTH);

    // D: rows over i (parallel), contiguous over l (vectorized)
    init_D
        .compute_root()
        .reorder(l, i)
        .parallel(i)
        .vectorize(l, VEC_WIDTH);

    init_A.realize(A);
    init_B.realize(B);
    init_C.realize(C);
    init_D.realize(D);
}

int main(int argc, char *argv[]) {
    // Problem sizes
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    // Scalar parameters
    num_t alpha;
    num_t beta;

    // Buffers corresponding to the C arrays:
    //
    // A[ni][nk] -> A(k, i)
    // B[nk][nj] -> B(j, k)
    // C[nj][nl] -> C(l, j)
    // D[ni][nl] -> D(l, i)
    // tmp[ni][nj] -> tmp(j, i)
    Halide::Buffer<num_t, 2> A(nk, ni);
    Halide::Buffer<num_t, 2> B(nj, nk);
    Halide::Buffer<num_t, 2> C(nl, nj);
    Halide::Buffer<num_t, 2> D(nl, ni);
    Halide::Buffer<num_t, 2> tmp(nj, ni);

    // ImageParams to pass the buffers into the Halide pipeline
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam D_param(type_of<num_t>(), 2, "D_param");

    // Initialize data (equivalent to init_array in the C version)
    init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);

    // Bind buffers to ImageParams
    A_param.set(A);
    B_param.set(B);
    C_param.set(C);
    D_param.set(D);

    // ------------------------------------------------------------------------
    // Halide definition of the 2mm kernel:
    //
    // Original C:
    //   /* D := alpha*A*B*C + beta*D */
    //   for (i)
    //     for (j)
    //       tmp[i][j] = 0;
    //       for (k)
    //         tmp[i][j] += alpha * A[i][k] * B[k][j];
    //
    //   for (i)
    //     for (j)
    //       D[i][j] *= beta;
    //       for (k)
    //         D[i][j] += tmp[i][k] * C[k][j];
    //
    // Mapping to Halide coordinates:
    //   i -> row index  (second dimension)
    //   j -> column index for tmp and D (first dimension)
    //   k -> reduction indices (nk for tmp, nj for D)
    //
    // Buffers:
    //   tmp(j, i) corresponds to tmp[i][j]
    //   A_param(k, i) corresponds to A[i][k]
    //   B_param(j, k) corresponds to B[k][j]
    //   C_param(l, j) corresponds to C[j][l]
    //   D_param(l, i) corresponds to D[i][l]
    // ------------------------------------------------------------------------

    Var i("i"), j("j"), l("l");
    RDom k_AB(0, A_param.dim(0).extent(), "k_AB");   // reduction over nk
    RDom k_DC(0, C_param.dim(1).extent(), "k_DC");   // reduction over nj

    Func tmp_func("tmp");
    Func D_func("D");

    // tmp[i][j] accumulation: tmp(j, i)
    tmp_func(j, i) = cast<num_t>(0);
    tmp_func(j, i) += Expr(alpha) * A_param(k_AB, i) * B_param(j, k_AB);

    // D[i][l] update: D(l, i)
    D_func(l, i) = Expr(beta) * D_param(l, i);
    D_func(l, i) += tmp_func(k_DC, i) * C_param(l, k_DC);

    // ------------------------------------------------------------------------
    // Scheduling
    //
    // Goals:
    //  - Preserve the mathematical result of the original 2mm kernel.
    //  - Exploit data locality:
    //      * Access contiguous x-dimension in vectors.
    //      * Keep inner loops over contiguous memory.
    //  - Use multi-core parallelism across rows (i dimension).
    //
    // Layout recap:
    //   tmp(j, i)  : x = j (contiguous), y = i
    //   D(l, i)    : x = l (contiguous), y = i
    //
    // Strategy:
    //   - For tmp:
    //       Pure:   reorder(j, i) so j is innermost; vectorize j; parallelize i.
    //       Update: reorder(j, k_AB, i) to get loops i (outer), k_AB (middle),
    //               j (innermost) and vectorize over j. This matches a GEMM-style
    //               i–k–j traversal: for each row i and k, process a vector of
    //               output columns j at once.
    //
    //   - For D:
    //       Pure:   reorder(l, i) so l is innermost; vectorize l; parallelize i.
    //       Update: reorder(l, k_DC, i) to get loops i (outer), k_DC (middle),
    //               l (innermost) and vectorize over l, again matching
    //               GEMM-style access patterns.
    //
    //   - Both Funcs are computed at root; the dependency tmp -> D is
    //     preserved by Halide automatically.
    // ------------------------------------------------------------------------

    // Compute both stages at root
    tmp_func.compute_root();
    D_func.compute_root();

    // ----- Schedule for tmp_func (A * B part) -----

    // Pure stage: initialize tmp(j, i) = 0
    //  i : outer (rows), j : inner (contiguous columns)
    tmp_func
        .reorder(j, i)              // innermost j, outermost i
        .parallel(i)                // distribute rows across threads
        .vectorize(j, VEC_WIDTH);   // process several columns per iteration

    // Update stage: tmp(j, i) += alpha * A(k_AB, i) * B(j, k_AB)
    // Desired loop order: for i (parallel), for k_AB, for j (vectorized)
    tmp_func.update()
        .reorder(j, k_AB, i)        // innermost j, then k_AB, then i
        .parallel(i)                // parallel over independent rows
        .vectorize(j, VEC_WIDTH);   // contiguous tmp/B dimension

    // This schedule yields (approximately):
    //   parallel_for i:
    //     for k_AB:
    //       for j in vectorized chunks:
    //         tmp[j, i] += alpha * A[k_AB, i] * B[j, k_AB];

    // ----- Schedule for D_func (tmp * C + beta*D part) -----

    // Pure stage: D(l, i) = beta * D_param(l, i)
    D_func
        .reorder(l, i)              // innermost l (contiguous), outermost i
        .parallel(i)                // parallel across rows of D
        .vectorize(l, VEC_WIDTH);   // vector over contiguous l

    // Update stage: D(l, i) += tmp(k_DC, i) * C(l, k_DC)
    // Desired loop order: for i (parallel), for k_DC, for l (vectorized)
    D_func.update()
        .reorder(l, k_DC, i)        // innermost l, then k_DC, then i
        .parallel(i)                // each row (i) is independent
        .vectorize(l, VEC_WIDTH);   // contiguous D/C dimension

    // ------------------------------------------------------------------------
    // JIT compile and run
    // ------------------------------------------------------------------------

    // Use the host target (possibly overridden by HL_TARGET/HL_JIT_TARGET).
    Target target = get_jit_target_from_environment();
    D_func.compile_jit(target);

    auto start = std::chrono::high_resolution_clock::now();

    // Run the kernel: result is written into D
    D_func.realize(D);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result matrix, similar to print_array in C.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < ni; ii++) {
            for (int jj = 0; jj < nl; jj++) {
                // D(ii, jj) in C is D(l = jj, i = ii) in Halide
                std::cerr << D(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}