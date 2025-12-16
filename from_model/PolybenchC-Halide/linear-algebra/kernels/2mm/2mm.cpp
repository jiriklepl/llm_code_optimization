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
 *
 *   Note: In all Buffers, the first dimension (x) is the fast-varying one.
 *   This matches C row-major layout when we map indices carefully.
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

    // C[j][l] = (DATA_TYPE) ((j*(l+3)+1) % nl) / nl;
    // C is stored as C(l, j) with shape (nl, nj)
    init_C(l, j) = cast<num_t>(((j * (l + 3) + 1) % nl)) / cast<int>(nl);

    // D[i][l] = (DATA_TYPE) (i*(l+2) % nk) / nk;
    // D is stored as D(l, i) with shape (nl, ni)
    init_D(l, i) = cast<num_t>(((i * (l + 2)) % nk)) / cast<int>(nk);

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
    //     for (j) {
    //       tmp[i][j] = 0;
    //       for (k)
    //         tmp[i][j] += alpha * A[i][k] * B[k][j];
    //     }
    //
    //   for (i)
    //     for (l) {
    //       D[i][l] *= beta;
    //       for (j)
    //         D[i][l] += tmp[i][j] * C[j][l];
    //     }
    //
    // Mapping to Halide coordinates:
    //   i -> row index of A, tmp, D (second dimension in Buffers)
    //   j -> column index of tmp and row index of C/B (first dimension)
    //   k -> reduction index over NK (A/B shared dim)
    //   l -> column index of C and D (first dimension)
    //
    // Buffers:
    //   tmp(j, i)        corresponds to tmp[i][j]
    //   A_param(k, i)    corresponds to A[i][k]
    //   B_param(j, k)    corresponds to B[k][j]
    //   C_param(l, j)    corresponds to C[j][l]
    //   D_param(l, i)    corresponds to D[i][l]
    // ------------------------------------------------------------------------

    Var i("i"), j("j"), l("l");
    RDom k_AB(0, A_param.dim(0).extent(), "k_AB");   // reduction over nk
    RDom k_DC(0, C_param.dim(1).extent(), "k_DC");   // reduction over nj

    Func tmp_func("tmp");
    Func D_func("D");

    // Stage 1: tmp[i][j] = alpha * sum_k A[i][k] * B[k][j]
    // tmp(j, i) = tmp[i][j]
    tmp_func(j, i) = cast<num_t>(0);
    tmp_func(j, i) += Expr(alpha) * A_param(k_AB, i) * B_param(j, k_AB);

    // Stage 2: D[i][l] = beta * D[i][l] + sum_j tmp[i][j] * C[j][l]
    // D(l, i) = D[i][l]
    D_func(l, i) = Expr(beta) * D_param(l, i);
    D_func(l, i) += tmp_func(k_DC, i) * C_param(l, k_DC);

    // ------------------------------------------------------------------------
    // Scheduling
    //
    // Goal:
    //   - Exploit data locality and SIMD in the fast-varying (x) dimension.
    //   - Parallelize across coarse tiles of rows/columns to leverage the
    //     many-core x64 CPU (32 physical cores).
    //   - Preserve original computation semantics.
    //
    // Memory layout recap (Halide Buffers):
    //   A(k, i):  A[i][k]  -> contiguous in k (dim 0)
    //   B(j, k):  B[k][j]  -> contiguous in j (dim 0)
    //   C(l, j):  C[j][l]  -> contiguous in l (dim 0)
    //   D(l, i):  D[i][l]  -> contiguous in l (dim 0)
    //   tmp(j, i):tmp[i][j]-> contiguous in j (dim 0)
    //
    // So we want to vectorize along:
    //   - j for tmp/B
    //   - l for C/D
    // and parallelize across tiles of (i, j) and (i, l).
    // ------------------------------------------------------------------------

    // Determine a reasonable SIMD width for the element type on this host.
    Target host_target = get_host_target();
    int vec_width = host_target.natural_vector_size(type_of<num_t>());
    if (vec_width < 1) {
        vec_width = 1;
    }

    // Tile sizes: chosen conservatively; can be tuned per architecture.
    const int tile_i = 32;
    const int tile_j = 32;
    const int tile_l = 32;
    const int tile_k_ab = 32;  // split factor for k_AB
    const int tile_k_dc = 32;  // split factor for k_DC

    // -----------------------------
    // Stage 1 schedule: tmp = A * B
    // -----------------------------
    tmp_func.compute_root();

    // Tile (j, i) into (j_outer, i_outer) tiles of size tile_j x tile_i.
    // Fuse tile indices into a single parallel dimension to avoid nested
    // parallel loops, then vectorize across j_inner (contiguous).
    Var j_outer("j_outer"), i_outer("i_outer");
    Var j_inner("j_inner"), i_inner("i_inner");
    Var tile_index_tmp("tile_index_tmp");

    tmp_func
        .tile(j, i, j_outer, i_outer, j_inner, i_inner, tile_j, tile_i)
        .fuse(j_outer, i_outer, tile_index_tmp)
        .parallel(tile_index_tmp)
        .vectorize(j_inner, vec_width);

    // Schedule the reduction update:
    // We iterate tiles in the same way, split k_AB into outer and inner parts,
    // and reorder loops so that:
    //   tile_index_tmp (parallel) ->
    //   k_outer ->
    //   i_inner ->
    //   k_inner ->
    //   j_inner (innermost, vectorized)
    RVar k_outer("k_outer"), k_inner("k_inner");

    tmp_func.update()
        .tile(j, i, j_outer, i_outer, j_inner, i_inner, tile_j, tile_i)
        .fuse(j_outer, i_outer, tile_index_tmp)
        .split(k_AB, k_outer, k_inner, tile_k_ab)
        // innermost to outermost:
        .reorder(j_inner, k_inner, i_inner, k_outer, tile_index_tmp)
        .parallel(tile_index_tmp)
        .vectorize(j_inner, vec_width)
        .unroll(k_inner, 4);  // small unroll for extra ILP

    // -----------------------------
    // Stage 2 schedule: D = tmp * C + beta * D
    // -----------------------------
    D_func.compute_root();

    // Tile (l, i) into (l_outer, i_outer2) tiles of size tile_l x tile_i.
    // Fuse tile indices for parallelism and vectorize across l_inner (contiguous).
    Var l_outer("l_outer"), i_outer2("i_outer2");
    Var l_inner("l_inner"), i_inner2("i_inner2");
    Var tile_index_D("tile_index_D");

    D_func
        .tile(l, i, l_outer, i_outer2, l_inner, i_inner2, tile_l, tile_i)
        .fuse(l_outer, i_outer2, tile_index_D)
        .parallel(tile_index_D)
        .vectorize(l_inner, vec_width);

    // Schedule the reduction update over k_DC:
    // Loop order:
    //   tile_index_D (parallel) ->
    //   k_outer2 ->
    //   i_inner2 ->
    //   k_inner2 ->
    //   l_inner (innermost, vectorized)
    RVar k_outer2("k_outer2"), k_inner2("k_inner2");

    D_func.update()
        .tile(l, i, l_outer, i_outer2, l_inner, i_inner2, tile_l, tile_i)
        .fuse(l_outer, i_outer2, tile_index_D)
        .split(k_DC, k_outer2, k_inner2, tile_k_dc)
        .reorder(l_inner, k_inner2, i_inner2, k_outer2, tile_index_D)
        .parallel(tile_index_D)
        .vectorize(l_inner, vec_width)
        .unroll(k_inner2, 4);

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