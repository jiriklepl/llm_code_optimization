#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "symm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Array initialization, translated from the original C version.
 *
 * Original C signature:
 *   void init_array(int m, int n,
 *                   DATA_TYPE *alpha,
 *                   DATA_TYPE *beta,
 *                   DATA_TYPE C[M][N],
 *                   DATA_TYPE A[M][M],
 *                   DATA_TYPE B[M][N]);
 *
 * Mapping:
 *   - C[i][j] (MxN) -> Buffer C(j, i) with shape (n, m)
 *   - B[i][j] (MxN) -> Buffer B(j, i) with shape (n, m)
 *   - A[i][j] (MxM) -> Buffer A(j, i) with shape (m, m)
 */
static
void init_array(int m, int n,
                num_t *alpha,
                num_t *beta,
                Halide::Buffer<num_t, 2> C,
                Halide::Buffer<num_t, 2> A,
                Halide::Buffer<num_t, 2> B) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j");

    // Initialize C and B:
    //
    // C[i][j] = (DATA_TYPE)((i + j) % 100) / m;
    // B[i][j] = (DATA_TYPE)((n + i - j) % 100) / m;
    //
    // With our Buffer layout C(j, i), B(j, i):
    Func init_C("init_C"), init_B("init_B");

    init_C(j, i) = cast<num_t>(((i + j) % 100)) / cast<int>(m);
    init_B(j, i) = cast<num_t>(((n + i - j) % 100)) / cast<int>(m);

    // Initialize A:
    //
    // for (i = 0; i < m; i++) {
    //   for (j = 0; j <= i; j++)
    //       A[i][j] = (DATA_TYPE)((i + j) % 100) / m;
    //   for (j = i+1; j < m; j++)
    //       A[i][j] = -999;
    // }
    //
    // We store A[i][j] at A(j, i), and fill the "unused" upper part with -999.
    Func init_A("init_A");
    Expr lower_tri = cast<num_t>(((i + j) % 100)) / cast<int>(m);
    Expr unused    = cast<num_t>(-999);
    init_A(j, i)   = select(j <= i, lower_tri, unused);

    // ------------------------------------------------------------------
    // Schedule for initialization:
    //
    //  - Parallelize over the slower-varying dimension i (rows, size m)
    //  - Vectorize over the faster-varying dimension j (columns, size n)
    //  - This gives good cache behavior and uses SIMD for all three inits.
    // ------------------------------------------------------------------
    const int vec = 8; // reasonable fixed vector width for modern x86
    init_C.compute_root()
          .parallel(i)
          .vectorize(j, vec);
    init_B.compute_root()
          .parallel(i)
          .vectorize(j, vec);
    init_A.compute_root()
          .parallel(i)
          .vectorize(j, vec);

    // Realize into the provided buffers
    init_C.realize(C);
    init_B.realize(B);
    init_A.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size (from symm.hpp / defines.hpp)
    int m = M;
    int n = N;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers:
    //   C, B : MxN  -> (n, m) in Halide
    //   A    : MxM  -> (m, m) in Halide
    Halide::Buffer<num_t, 2> C(n, m);
    Halide::Buffer<num_t, 2> B(n, m);
    Halide::Buffer<num_t, 2> A(m, m);

    // ImageParams to pass the buffers into the Halide pipeline
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");

    // Initialize data
    init_array(m, n, &alpha, &beta, C, A, B);

    // Vars and reduction domain
    Var i("i"), j("j");
    // A is MxM stored as A(j, i); extent of dim(0) is m
    RDom k(0, A.dim(0).extent(), "k");

    // Main computational kernel in Halide.
    //
    // Original BLAS semantics:
    //   C := alpha * A * B + beta * C
    // with A symmetric, stored in its lower triangular part.
    //
    // We construct a symmetric view of A:
    //   A_sym(i, l) =
    //      (l <= i) ? A(i, l)  : A(l, i)
    //
    // Remember our Buffer mapping:
    //   original A[i][l] -> A_param(l, i)
    //   original A[l][i] -> A_param(i, l)
    //
    // So we define:
    //   A_sym(i, k) = (k <= i) ? A_param(k, i) : A_param(i, k)
    //
    // Then:
    //   symm(j, i) = beta * C_param(j, i)
    //   symm(j, i) += alpha * sum_k A_sym(i, k) * B_param(j, k)
    //
    Func symm("symm");

    // Base: beta * C
    symm(j, i) = Expr(beta) * C_param(j, i);

    // Symmetric access to A
    Expr A_sym = select(k <= i, A_param(k, i), A_param(i, k));

    // Accumulate alpha * A_sym * B
    symm(j, i) += Expr(alpha) * A_sym * B_param(j, k);

    // ------------------------------------------------------------------
    // Schedule for the SYMM kernel
    //
    // Layout recap:
    //   - C and B have shape (n, m) = (j, i)
    //   - A has shape (m, m)       = (k, i or i, k)
    //
    // We want:
    //   - j (columns, dim 0) to be the innermost vectorized dimension
    //     to get unit-stride access into C and B.
    //   - Parallelism across 2D tiles of (j, i) using all cores.
    //   - A reduction over k for each (j, i), with k outside the
    //     innermost vectorized j loop, so we reuse B(j, k) in-register
    //     across the vector.
    //
    // Tiling:
    //   tile size in j  (fast dim) = 64
    //   tile size in i  (slow dim) = 32
    //
    // We then fuse the tile coordinates and parallelize across tiles.
    // ------------------------------------------------------------------
    Var jo("jo"), ji("ji"), io("io"), ii("ii"), tile("tile");
    const int tile_j = 64;
    const int tile_i = 32;
    const int vec    = 8;   // SIMD width along j

    // Pure definition (initialization with beta * C)
    symm
        .tile(j, i, jo, io, ji, ii, tile_j, tile_i)
        .fuse(jo, io, tile)   // 2D tiles -> 1D index for parallelism
        .parallel(tile)       // distribute tiles across cores
        .vectorize(ji, vec);  // vectorize along contiguous j

    // Update definition (GEMM-like accumulation)
    //
    // After tiling/fusing, the loop order before reordering is:
    //   for tile
    //     for ii
    //       for ji
    //         for k
    //
    // That makes k the innermost loop, which is bad for B(j, k) since
    // it would step with a large stride in memory. We want j to be the
    // innermost loop, so we reorder to:
    //   for tile (outermost)
    //     for ii
    //       for k
    //         for ji (innermost, vectorized)
    //
    // This keeps j contiguous and SIMD-friendly while we accumulate
    // across k.
    symm.update()
        .reorder(ji, k, ii, tile)
        .vectorize(ji, vec);

    // Bind buffers to ImageParams
    C_param.set(C);
    A_param.set(A);
    B_param.set(B);

    // JIT-compile the kernel
    symm.compile_jit();

    // Time the kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write result back into C
    symm.realize(C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Print results (mainly to prevent dead-code elimination),
    // following the style of the GEMM example.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < m; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // C(ii, jj) in C layout is C(jj, ii) in our Buffer
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}