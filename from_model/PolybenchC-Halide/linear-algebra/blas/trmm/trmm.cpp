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
 * Original C version:
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
 * In this optimized Halide version we choose a memory layout that is
 * friendlier to the TRMM access patterns:
 *
 *   - Conceptually, A is still MxM with the same values as the C code.
 *
 *   - We store A in a Buffer A(w = m, h = m) but interpret the coordinates
 *     as A(k, i) == A[k][i] from the C code:
 *         k : row index  (0..m-1)
 *         i : column index (0..m-1)
 *     Dimension 0 (k) is unit-stride in memory, so accesses A(k, i) with
 *     varying k are contiguous. This matches the kernel’s usage pattern
 *     A[k][i] with k as the “inner” index.
 *
 *   - B is MxN in the original C code, accessed as B[i][j]. We keep the
 *     same row-major logical layout by storing it as Buffer B(width = n,
 *     height = m) and accessing it as B(j, i) == B[i][j]:
 *         j : column index (0..n-1)  -> unit-stride in memory
 *         i : row index    (0..m-1)
 */
static void init_array(int m, int n,
                       num_t *alpha,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B) {
    *alpha = static_cast<num_t>(1.5);

    Var i("i"), k("k"), j("j");
    Func init_A("init_A"), init_B("init_B");

    // Make Expr-typed versions of m and n for use inside Halide expressions.
    Expr m_expr = Expr(m);
    Expr n_expr = Expr(n);

    // A(k, i) corresponds to C's A[k][i]:
    //
    //   for each row k:
    //     for each column i:
    //       if (i < k)      A[k][i] = (k + i) % m / m
    //       else if i == k  A[k][i] = 1.0
    //       else            A[k][i] = 0.0   (never used by TRMM, but defined)
    //
    // This matches the original lower-triangular-with-unit-diagonal pattern,
    // just with "row = k, col = i".
    init_A(k, i) =
        select(i < k,
               cast<num_t>(((k + i) % m_expr)) / cast<num_t>(m_expr),
               select(i == k,
                      Expr(static_cast<num_t>(1.0)),
                      Expr(static_cast<num_t>(0.0))));

    // B(j, i) corresponds to C's B[i][j]:
    //
    //   B[i][j] = ( (n + (i - j)) % n ) / n
    //
    // We keep the same formula but map (i,j) -> B(j, i).
    init_B(j, i) =
        cast<num_t>(((n_expr + (i - j)) % n_expr)) / cast<num_t>(n_expr);

    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem sizes (compile-time constants from PolyBench headers)
    int m = M;
    int n = N;

    // Scalars and arrays
    num_t alpha;

    // Buffer layout:
    //
    // A: M x M
    //   - Buffer A(width=m, height=m)
    //   - A(k, i) == A[k][i] in the original C code
    //   - k (dim 0) is unit-stride -> good for A[k, i] with k varying
    //
    // B: M x N
    //   - Buffer B(width=n, height=m)
    //   - B(j, i) == B[i][j] in the original C code
    //   - j (dim 0) is unit-stride -> good for iterating across columns j
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

    // Reduction domain over k in [0, m)
    RDom k(0, m, "k");

    // -------------------------------------------------------------------------
    // Triangular matrix-matrix multiply:
    //
    // Original C kernel (TRMM variant, using 0-based indices):
    //
    //   for (i = 0; i < M; i++)
    //     for (j = 0; j < N; j++) {
    //       for (k = i+1; k < M; k++)
    //         B[i][j] += A[k][i] * B[k][j];
    //       B[i][j] = alpha * B[i][j];
    //     }
    //
    // Because all reads of B[k][j] use rows k > i which have not yet been
    // updated by the outer i-loop, every final B[i][j] depends only on the
    // *original* B values. We can therefore express the closed-form result:
    //
    //   B_out[i][j] = alpha * ( B_in[i][j] +
    //                           sum_{k = i+1..M-1} A[k][i] * B_in[k][j] )
    //
    // Mapping indices to Halide coordinates:
    //
    //   A[k][i]  -> A_param(k, i)
    //   B[k][j]  -> B_param(j, k)
    //   B[i][j]  -> B_param(j, i)
    //
    // We implement the triangular domain "k from i+1..M-1" using a predicate
    // inside a reduction over a simple rectangular RDom k in [0..M-1].
    // -------------------------------------------------------------------------

    Func trmm("trmm");

    Type t = type_of<num_t>();
    Expr alpha_expr = cast(t, Expr(alpha));
    Expr zero = make_zero(t);

    // Base value: alpha * B_in[i][j]
    //   B_in[i][j] -> B_param(j, i)
    trmm(j, i) = alpha_expr * B_param(j, i);

    // Reduction: accumulate alpha * A[k][i] * B[k][j] for k > i
    //
    //   A[k][i]  -> A_param(k, i)
    //   B[k][j]  -> B_param(j, k)
    //
    // We guard the multiply with (k > i) to enforce the triangular domain.
    trmm(j, i) += alpha_expr *
                  select(k > i,
                         A_param(k, i) * B_param(j, k),
                         zero);

    // -------------------------------------------------------------------------
    // Schedule
    //
    // Goals:
    //   - Treat TRMM as a triangular variant of GEMM.
    //   - Improve cache locality via blocking in i and j.
    //   - Exploit parallelism over tiles of rows (i).
    //   - Vectorize along j (contiguous dimension in B).
    //   - Keep the k-reduction register-resident and unrolled.
    // -------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();
    int vec_width = target.natural_vector_size(t);
    if (vec_width < 1) {
        vec_width = 1;
    }

    // Tile sizes chosen conservatively for a modern multi-core x64 CPU.
    const int tile_i = 64;   // rows of B / columns of A
    const int tile_j = 128;  // columns of B
    const int unroll_k = 4;  // unroll factor for k inside tiles

    Var io("io"), jo("jo"), ii("ii"), jj("jj");
    RVar ko("ko"), ki("ki");

    // Pure definition: tile (j, i), parallelize over tiles of i, and
    // vectorize over j inside tiles. This defines the loop nest for
    // initializing trmm(j, i) = alpha * B(j, i).
    trmm
        .tile(j, i, jo, io, jj, ii,
              tile_j, tile_i, TailStrategy::GuardWithIf)
        // Innermost to outermost: jj (SIMD over columns), ii (rows in tile),
        // then tile indices jo, io.
        .reorder(jj, ii, jo, io)
        .vectorize(jj, vec_width)
        .parallel(io);

    // Update definition (reduction over k): we use the same tiling in (j, i),
    // split k for unrolling, and order loops as:
    //
    //   for io (parallel tiles of rows)
    //     for jo (tiles of cols)
    //       for ko (chunks of k)
    //         for ii (rows in tile)
    //           for ki (unrolled k within chunk)
    //             for jj (vectorized columns)
    //
    // This structure matches a blocked, BLAS-3-like TRMM:
    //   - For each tile of rows io and cols jo,
    //   - For each k-chunk ko, we touch a panel of A(:, i) and a panel of
    //     B(:, j), and accumulate into a tile of trmm.
    trmm.update()
        .tile(j, i, jo, io, jj, ii,
              tile_j, tile_i, TailStrategy::GuardWithIf)
        .split(k, ko, ki, unroll_k)
        .reorder(jj, ki, ii, ko, jo, io)
        .vectorize(jj, vec_width)
        .unroll(ki)
        .parallel(io);

    // Compile the kernel to machine code
    trmm.compile_jit(target);

    // Output buffer for the result (same logical layout as B)
    Buffer<num_t, 2> B_result(n, m);

    // Time the execution
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