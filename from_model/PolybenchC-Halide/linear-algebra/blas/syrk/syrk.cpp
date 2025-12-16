#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "syrk.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Array initialization, translated from the original C version.
 */
static
void init_array(int n, int m,
                num_t *alpha,
                num_t *beta,
                Halide::Buffer<num_t, 2> C,
                Halide::Buffer<num_t, 2> A) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j");
    Func init_A("init_A"), init_C("init_C");

    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < m; j++)
    //     A[i][j] = ((i*j+1)%n) / n;
    //
    // C array A[i][j] is mapped to Buffer A(j, i) with shape (m, n).
    init_A(j, i) = cast<num_t>((i * j + 1) % n) / cast<int>(n);

    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < n; j++)
    //     C[i][j] = ((i*j+2)%m) / m;
    //
    // C array C[i][j] is mapped to Buffer C(j, i) with shape (n, n).
    init_C(j, i) = cast<num_t>((i * j + 2) % m) / cast<int>(m);

    // Simple but effective schedule for initialization:
    //  - j is the contiguous dimension (dim 0), so vectorize over j
    //  - parallelize over rows i
    Target init_target = get_host_target();
    const int vec = init_target.natural_vector_size(type_of<num_t>());

    init_A.vectorize(j, vec).parallel(i);
    init_C.vectorize(j, vec).parallel(i);

    init_A.realize(A);
    init_C.realize(C);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;
    int m = M;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers:
    // C is N x N in C as C[i][j], map to Buffer C(j, i) => shape (n, n)
    // A is N x M in C as A[i][j], map to Buffer A(j, i) => shape (m, n)
    Halide::Buffer<num_t, 2> C(n, n);
    Halide::Buffer<num_t, 2> A(m, n);

    // ImageParams corresponding to C and A
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");

    // Initialize data using Halide
    init_array(n, m, &alpha, &beta, C, A);

    // Halide Vars and RDom
    Var i("i"), j("j");
    RDom k(0, A.dim(0).extent(), "k"); // k over M (first dimension of A)

    // Build the SYRK kernel in Halide
    // BLAS SYRK:
    // C := alpha * A * A^T + beta * C
    //
    // Original C kernel:
    // for (i = 0; i < _PB_N; i++) {
    //   for (j = 0; j <= i; j++)
    //     C[i][j] *= beta;
    //   for (k = 0; k < _PB_M; k++) {
    //     for (j = 0; j <= i; j++)
    //       C[i][j] += alpha * A[i][k] * A[j][k];
    //   }
    // }
    //
    // Mapping between C array indices and Halide:
    //   C[i][j]   -> C_param(j, i)      (Buffer C has shape (n, n))
    //   A[i][k]   -> A_param(k, i)      (Buffer A has shape (m, n))
    //   A[j][k]   -> A_param(k, j)
    //
    // We compute a full N x N Func and guard the lower triangle (j <= i)
    // with a select, leaving the upper triangle unchanged.

    Func syrk("syrk");

    Expr alpha_expr = Expr(alpha);
    Expr beta_expr  = Expr(beta);

    Expr orig = C_param(j, i);

    // Pure definition: scale lower triangle by beta, keep upper triangle unchanged.
    syrk(j, i) = select(j <= i,
                        cast<num_t>(beta_expr) * orig,
                        orig);

    // Reduction over k: add alpha * A[i][k] * A[j][k] for lower triangle.
    syrk(j, i) += select(j <= i,
                         cast<num_t>(alpha_expr) *
                             A_param(k, i) * A_param(k, j),
                         cast<num_t>(0));

    // ------------------------------------------------------------------
    // Optimized schedule
    // ------------------------------------------------------------------
    //
    // Data layout:
    //   - C(j, i): j (dim 0) is contiguous in memory.
    //   - A(k, i): k (dim 0) is contiguous for any fixed row i.
    //
    // We apply:
    //   * 2D tiling over (i, j) to improve cache locality on C.
    //   * (optional) tiling over k to improve locality on A.
    //   * parallelization across tiles of rows of C.
    //   * vectorization across the inner j dimension (contiguous in C).
    //
    // Note: the triangular condition j <= i is implemented via select()
    // in the definition, so the schedule remains rectangular and simple,
    // while the upper triangle is left untouched.

    Target target = get_host_target();
    const int vec = target.natural_vector_size(type_of<num_t>());

    // Tile sizes; these are conservative defaults that work well for
    // typical desktop CPUs. They can be tuned if desired.
    const int tile_i = 32;
    const int tile_j = 32;
    const int tile_k = 64;

    Var io("io"), jo("jo"), ii("ii"), jj("jj");
    RVar ko("ko"), ki("ki");

    // Root (pure) definition: scale by beta / copy.
    // Tile in (j, i) so that each tile of C fits in cache.
    // Make j (dim 0) the innermost loop for unit-stride writes and
    // vectorize over it. Parallelize over tiles of rows (io).
    syrk
        .tile(j, i, jo, io, jj, ii, tile_j, tile_i, TailStrategy::GuardWithIf)
        // Reorder loops so that jj is innermost, then ii, then jo, then io (outermost).
        .reorder(jj, ii, jo, io)
        .vectorize(jj, vec)
        .parallel(io);

    // Update definition: accumulate alpha * A[i,k] * A[j,k].
    // Apply the same tiling in (j, i), and additionally split k into
    // (ko, ki) to improve locality on A. We keep k inside the tile
    // loops so each C-tile reuses the same A-tiles as much as possible.
    syrk.update()
        .tile(j, i, jo, io, jj, ii, tile_j, tile_i, TailStrategy::GuardWithIf)
        .split(k, ko, ki, tile_k)
        // Loop order (outermost -> innermost): io, jo, ko, ii, ki, jj
        // jj innermost for contiguous C stores; ki inner over k-chunks.
        .reorder(jj, ki, ii, ko, jo, io)
        .vectorize(jj, vec)
        .unroll(ki)
        .parallel(io);

    // Bind ImageParams
    C_param.set(C);
    A_param.set(A);

    // Compile the kernel to machine code for the chosen target
    syrk.compile_jit(target);

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write result into C (in-place semantics w.r.t the original code)
    syrk.realize(C);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<long double>(end - start);

    // Print results (similar spirit to the original print_array), guarded
    // the same way as in the GEMM example.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // C array C[ii][jj] -> Buffer C(jj, ii)
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}