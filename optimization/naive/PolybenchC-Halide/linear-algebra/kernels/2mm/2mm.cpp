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

    // A simple but reasonably efficient schedule: parallelize the outer
    // dimension and vectorize the inner one. This affects only
    // initialization cost, not the kernel itself.
    const int vec_width = 8; // reasonable default on modern x86

    init_A
        .compute_root()
        .parallel(i)
        .vectorize(k, vec_width);
    init_B
        .compute_root()
        .parallel(j)
        .vectorize(k, vec_width);
    init_C
        .compute_root()
        .parallel(j)
        .vectorize(l, vec_width);
    init_D
        .compute_root()
        .parallel(i)
        .vectorize(l, vec_width);

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
    //   - Preserve exact numerical behavior of the original 2mm kernel.
    //   - Improve data locality and cache reuse.
    //   - Exploit parallelism across rows (i) and SIMD within rows/columns.
    //
    // Key idea:
    //   The original C code computes tmp row-by-row and then uses that row
    //   to update the corresponding row of D. We mimic that structure by
    //   computing tmp_func *inside* the loop over i in D_func, so that:
    //
    //     for i:
    //       compute tmp(i, all j)
    //       compute D(i, all l) using that tmp row
    //
    //   This avoids storing the entire tmp matrix at once, improves temporal
    //   locality, and still matches the mathematical definition exactly.
    //
    // Parallelism:
    //   - Each row i of the final D is independent ==> parallelize over i.
    //   - Within a row, l (for D) and j (for tmp) are contiguous in memory
    //     ==> vectorize over those dimensions.
    // ------------------------------------------------------------------------

    const int vec_width = 8; // reasonable default on modern x86; Halide handles tails.

    // Schedule for the final result D:
    //
    // Layout of D: D(l, i) with l as x-dimension (contiguous) and i as y.
    // Default loop nest for compute_root would be:
    //   for i
    //     for l
    // which already matches the original C order (i outer, l inner).
    //
    // We:
    //   - compute D at root,
    //   - parallelize over i (rows),
    //   - vectorize over l (columns).
    D_func
        .compute_root()
        .parallel(i)
        .vectorize(l, vec_width);

    // Schedule for tmp:
    //
    // We want to compute each row of tmp right before we compute the
    // corresponding row of D. That is, we "fuse" the original two C
    // loops over i by computing tmp at the loop level 'i' of D_func.
    //
    // This gives a loop structure equivalent to:
    //
    //   for i in [0..ni):
    //     compute tmp(j, i) for all j  (using reduction over k_AB)
    //     compute D(l, i) for all l    (pure + reduction over k_DC)
    //
    // Halide will allocate storage for tmp only for the slice needed
    // at this 'i', so memory usage and cache locality are both improved.
    tmp_func
        .compute_at(D_func, i)   // per row of D
        .reorder(j, i)           // j (x) inner, i (y) outer
        .vectorize(j, vec_width);

    // tmp's update definition: sum over k_AB for each (j, i).
    // Reorder so that the reduction over k_AB is innermost, with j
    // just inside i. Keep the same parallel structure as the pure stage
    // (no extra parallelization here; parallelism comes from D's 'i').
    tmp_func
        .update()
        .reorder(k_AB, j, i)
        .vectorize(j, vec_width);

    // Schedule for D's update definition:
    //
    // D(l, i) += tmp(k_DC, i) * C(l, k_DC);
    //
    // We want:
    //   - i outer, parallel
    //   - l inner, vectorized
    //   - k_DC as the innermost reduction loop.
    D_func
        .update()
        .reorder(k_DC, l, i)
        .parallel(i)
        .vectorize(l, vec_width);

    // ------------------------------------------------------------------------
    // JIT compile and run
    // ------------------------------------------------------------------------

    D_func.compile_jit();

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