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
    // Mapping in this file:
    //   C[i][j]     -> C(j, i)      with shape (n, n)
    //   A[i][j]     -> A(j, i)      with shape (m, n)
    //   B[i][j]     -> B(j, i)      with shape (m, n)

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
    // We map:
    //   C[i][j]   -> C(j, i) with shape (n, n)
    //   A[i][j]   -> A(j, i) with shape (m, n)
    //   B[i][j]   -> B(j, i) with shape (m, n)
    //
    // Dimension 0 is the fast-varying (contiguous) dimension.
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
    // A.dim(0) == m, so k ranges over the shared inner dimension.
    RDom k(0, A.dim(0).extent(), "k");

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
    // We compute on the full rectangular domain [0..n-1]x[0..n-1],
    // and use a select(j <= i, ...) to restrict updates to the lower
    // triangle. For j > i, C[i][j] is simply copied through.

    Func syr2k("syr2k");

    // Mapping between original C arrays and ImageParams:
    //
    //   C[i][j]   -> C_param(j, i)
    //   A[j][k]   -> A_param(k, j)
    //   B[i][k]   -> B_param(k, i)
    //   B[j][k]   -> B_param(k, j)
    //   A[i][k]   -> A_param(k, i)
    //
    // with A and B having shape (m, n): dim(0) = k in [0..m), dim(1) = row index in [0..n).

    Expr alpha_e = Expr(alpha);
    Expr beta_e  = Expr(beta);

    // Step 1: scale the lower triangle by beta; upper triangle is copied unchanged.
    syr2k(j, i) = select(j <= i,
                         beta_e * C_param(j, i),
                         C_param(j, i));

    // Step 2: symmetric rank‑2k update on the lower triangle:
    //   C[i][j] += alpha*(A[j][k]*B[i][k] + B[j][k]*A[i][k])  for j <= i
    //
    // Implemented as a reduction over k with a guard on (j <= i).
    syr2k(j, i) += select(
        j <= i,
        alpha_e * (A_param(k, j) * B_param(k, i) +
                   B_param(k, j) * A_param(k, i)),
        cast<num_t>(0));

    // -----------------------
    // Scheduling (optimization)
    // -----------------------
    //
    // Goals:
    //   * Keep j (the column index, mapped to buffer dim 0) as the
    //     innermost loop to get unit-stride accesses to C.
    //   * Tile the (i,j) space and the reduction in k to improve cache
    //     locality for A, B and C.
    //   * Parallelize across rows of C (i dimension) to exploit many cores.
    //   * Vectorize along j for SIMD.
    //
    // Note: we still use the j <= i guard to represent the triangular
    // structure; all tiling/reordering preserves this condition.

    // Machine-specific vector width for num_t on the host.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Tile sizes – empirically reasonable defaults for modern x64.
    const int tile_i = 32;   // rows of C per tile
    const int tile_j = 32;   // cols of C per tile
    const int tile_k = 32;   // reduction extent per tile

    // Tiled and split variables.
    Var i_outer("i_outer"), i_inner("i_inner");
    Var j_outer("j_outer"), j_inner("j_inner");
    RVar k_outer("k_outer"), k_inner("k_inner");

    // ---- Schedule for the pure definition (scaling / copy of C) ----
    //
    // Loop structure after tiling and reordering:
    //   for i_outer in tiles of size tile_i (parallel):
    //     for j_outer in tiles of size tile_j:
    //       for i_inner in [0, tile_i):
    //         for j_inner in [0, tile_j) (vectorized):
    //           syr2k(j, i) = (j <= i ? beta*C : C)
    syr2k
        .tile(j, i, j_outer, i_outer, j_inner, i_inner, tile_j, tile_i)
        // Reorder so that:
        //   outermost:  i_outer (rows, good for parallelism)
        //   then:       j_outer (column tiles)
        //   then:       i_inner
        //   innermost:  j_inner (unit-stride, good for SIMD)
        .reorder(j_inner, i_inner, j_outer, i_outer)
        .vectorize(j_inner, vec_width)
        .parallel(i_outer);

    // ---- Schedule for the reduction update (rank-2k update) ----
    //
    // Desired high-level loop structure:
    //
    //   for i_outer in tiles of i (parallel):
    //     for j_outer in tiles of j:
    //       for k_outer in tiles of k:
    //         for i_inner in tile_i:
    //           for k_inner in tile_k:
    //             for j_inner in tile_j (vectorized):
    //               if (j <= i) syr2k(j, i) += ...
    //
    // This keeps:
    //   * i_outer outermost for coarse-grain parallelism.
    //   * j_inner innermost for unit-stride SIMD stores to C.
    //   * A and B reused from cache within k_outer tiles.

    syr2k.update()
        // Tile over (j, i) exactly as for the pure definition.
        .tile(j, i, j_outer, i_outer, j_inner, i_inner, tile_j, tile_i)
        // Split the reduction over k into tiles of size tile_k.
        .split(k, k_outer, k_inner, tile_k)
        // Reorder loops from innermost outwards:
        //   innermost:   j_inner  (vectorized)
        //   then:        k_inner  (small reduction chunk)
        //   then:        k_outer  (reduction tiles)
        //   then:        i_inner
        //   then:        j_outer
        //   outermost:   i_outer  (parallel)
        .reorder(j_inner, k_inner, k_outer, i_inner, j_outer, i_outer)
        .vectorize(j_inner, vec_width)
        .unroll(k_inner)          // unroll small inner k chunk for ILP
        .parallel(i_outer);

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
                // Recall: C(ii,jj) in the original C mapping corresponds
                // to C(jj,ii) in the Buffer layout.
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}