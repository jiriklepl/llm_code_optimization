#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common Polybench definitions (NI, NJ, NK, NL, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions for 2mm
#include "2mm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize arrays A, B, C, D in the same way as the original C version.
 *
 * C layout:
 *   A[ni][nk] -> Buffer A(k, i)  with shape (nk, ni)
 *   B[nk][nj] -> Buffer B(j, k)  with shape (nj, nk)
 *   C[nj][nl] -> Buffer C(l, j)  with shape (nl, nj)
 *   D[ni][nl] -> Buffer D(l, i)  with shape (nl, ni)
 */
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

    // A[i][k] = (DATA_TYPE) ((i*k+1) % ni) / ni;
    // A is stored as A(k, i) with shape (nk, ni)
    init_A(k, i) = cast<num_t>(((i * k + 1) % ni)) / cast<int>(ni);

    // B[k][j] = (DATA_TYPE) (k*(j+1) % nj) / nj;
    // B is stored as B(j, k) with shape (nj, nk)
    init_B(j, k) = cast<num_t>(((k * (j + 1)) % nj)) / cast<int>(nj);

    // C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
    // C is stored as C(l, j) with shape (nl, nj)
    init_C(l, j) = cast<num_t>(((j * (l + 3) + 1) % nl)) / cast<int>(nl);

    // D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
    // D is stored as D(l, i) with shape (nl, ni)
    init_D(l, i) = cast<num_t>(((i * (l + 2)) % nk)) / cast<int>(nk);

    // ----------------------------
    // Initialization scheduling
    // ----------------------------
    // Use a reasonable vector width for the host and DATA_TYPE.
    Target target = get_host_target();
    const int vec = natural_vector_size<num_t>(target);

    // A: shape (nk, ni) as (k, i). Vectorize along contiguous k, parallelize rows in i.
    init_A
        .compute_root()
        .reorder(k, i)         // i outer, k inner
        .vectorize(k, vec)
        .parallel(i);

    // B: shape (nj, nk) as (j, k). j is contiguous.
    init_B
        .compute_root()
        .reorder(j, k)         // k outer, j inner
        .vectorize(j, vec)
        .parallel(k);

    // C: shape (nl, nj) as (l, j). l is contiguous.
    init_C
        .compute_root()
        .reorder(l, j)         // j outer, l inner
        .vectorize(l, vec)
        .parallel(j);

    // D: shape (nl, ni) as (l, i). l is contiguous.
    init_D
        .compute_root()
        .reorder(l, i)         // i outer, l inner
        .vectorize(l, vec)
        .parallel(i);

    // Realize into the provided buffers
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
    //
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
    // High-level goals:
    //   * Parallelize over i (rows).
    //   * Vectorize over the inner, contiguous dimensions (j for tmp, l for D).
    //   * Improve locality by computing a single row of tmp on-demand
    //     for each i used in D, instead of materializing the whole tmp.
    //
    // Data layout:
    //   * All Buffers are in default Halide layout: dim 0 is contiguous.
    //   * For tmp_func(j, i): j (dim 0) is contiguous.
    //   * For D_func(l, i) and D_param(l, i): l (dim 0) is contiguous.
    //   * For B_param(j, k): j (dim 0) is contiguous.
    //   * For C_param(l, k): l (dim 0) is contiguous.
    //
    // tmp_func schedule:
    //   - We compute tmp per row i inside the computation of D (compute_at).
    //   - Loops are ordered as: i (outer), k_AB (middle), j (inner, vectorized).
    //     This gives:
    //       * Inner loop over j -> contiguous access to B(j, k_AB).
    //       * A_param(k_AB, i) is constant across j and reused from registers.
    //
    // D_func schedule:
    //   - Compute at root.
    //   - Parallelize over i (rows).
    //   - Vectorize over l (columns).
    //   - Update loop ordered as: i (outer), k_DC (middle), l (inner, vectorized).
    //     This gives:
    //       * Inner loop over l -> contiguous access to C(l, k_DC) and D(l, i).
    //       * tmp_func(k_DC, i) reused across the inner vector of l.
    // ------------------------------------------------------------------------

    Target target = get_host_target();
    const int vec = natural_vector_size<num_t>(target);

    // Final output D: pure definition
    D_func
        .compute_root()
        .reorder(l, i)          // i outer, l inner (contiguous)
        .vectorize(l, vec)
        .parallel(i);

    // D update (reduction over k_DC)
    D_func.update()
        // We want loops: for i (parallel), for k_DC, for l (vectorized, contiguous)
        .reorder(l, k_DC, i)
        .vectorize(l, vec)
        .parallel(i);

    // tmp: compute one row per i inside D's outer loop.
    tmp_func
        // Allocate and store tmp rows at the level of i in D.
        // This limits the storage to a single row (length nj), improving locality.
        .store_at(D_func, i)
        .compute_at(D_func, i)
        .reorder(j, i)          // i outer, j inner (contiguous)
        .vectorize(j, vec);

    // tmp update (reduction over k_AB)
    tmp_func.update()
        // For each i, we run: for k_AB, for j (vectorized)
        // That is, loops: i (outer), k_AB (middle), j (inner).
        // j inner gives contiguous, vectorizable access to B(j, k_AB),
        // while A_param(k_AB, i) is reused across the j loop.
        .reorder(j, k_AB, i)
        .vectorize(j, vec);

    // ------------------------------------------------------------------------
    // JIT compile and run
    // ------------------------------------------------------------------------

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