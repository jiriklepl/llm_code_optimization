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
 *
 * Note: The layout for B corresponds to the original row-major B[k][j]
 *       (k is the row, j is the column): physical index is k*NJ + j.
 *       We store it as B(j, k), so B(j, k) == B_in[k][j], and the
 *       j-dimension is the contiguous (stride‑1) dimension. This is
 *       friendly for vectorization along j.
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
    // Buffer layout: B(j, k) where original i is our k, and j is j
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
    //   C: NI x NJ => (width = nj, height = ni)  stored as C(j, i)
    //   A: NI x NK => (width = nk, height = ni)  stored as A(k, i)
    //   B: NK x NJ => (width = nj, height = nk)  stored as B(j, k)
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

    // -------------------------------------------------------------------------
    // Optimized schedule for GEMM
    //
    // Goals:
    //   - Preserve numerical semantics of the PolyBench GEMM kernel.
    //   - Improve data locality via tiling in (i, j).
    //   - Exploit multiple cores via parallelism over tiles.
    //   - Exploit SIMD via vectorization along the contiguous j dimension.
    //   - Reduce loop overhead and increase ILP via unrolling in k.
    //
    // Memory layout summary:
    //   - C(j, i): C[i][j] in row-major, j is stride-1.
    //   - A(k, i): A[i][k] in row-major, k is stride-1.
    //   - B(j, k): B[k][j] in row-major, j is stride-1.
    //
    // We keep j as the innermost spatial dimension to get unit-stride
    // accesses to C and B, and we reuse A across the inner j loop.
    // -------------------------------------------------------------------------

    // Query a reasonable vector width for the current target and element type.
    Target target = get_host_target();
    const int vec = target.natural_vector_size(type_of<num_t>());

    // Tile sizes; chosen to be cache-friendly and to expose enough parallel work.
    const int tile_j = 64;  // columns of C (fast dimension)
    const int tile_i = 32;  // rows of C

    // Tile indices and intra-tile indices.
    Var io("io"), jo("jo"), ii("ii"), ji("ji");

    // Explicitly compute gemm at the root so we can tile/parallelize it.
    gemm.compute_root();

    // --- Schedule for the pure definition (C := beta * C) ---
    //
    // We tile over (j, i) so that each tile of C has size tile_i x tile_j.
    // Within a tile:
    //   - io, jo index the outer tiles
    //   - ii, ji index rows/cols inside the tile
    //
    // We:
    //   - reorder to make ji innermost and io outermost,
    //   - parallelize across io (tiles in the i dimension),
    //   - vectorize across ji (columns; stride-1).
    gemm
        .tile(j, i,  // original variables
              jo, io, // outer tile indices: jo over columns, io over rows
              ji, ii, // inner intra-tile indices: ji over columns, ii over rows
              tile_j, tile_i,
              TailStrategy::GuardWithIf)
        // From inner to outer: ji (vectorized), ii, jo, io
        .reorder(ji, ii, jo, io)
        .parallel(io)
        .vectorize(ji, vec);

    // --- Schedule for the reduction update (C += alpha * A * B) ---
    //
    // We use the same tiling in (j, i) so that the update passes over C
    // in the same cache-friendly order. For each (io, jo, ii, ji) tile
    // position, we iterate k over [0, nk) and accumulate:
    //
    //   for io (tiles of rows)
    //     for jo (tiles of cols)
    //       for ii (rows within tile)
    //         for k
    //           for ji (cols within tile, innermost, vectorized)
    //
    // This keeps:
    //   - C(i,j): stride-1 stores along ji.
    //   - B(j,k): stride-1 loads along ji for each fixed k.
    //   - A(k,i): reused across ji for each fixed (i,k).
    //
    // We also:
    //   - parallelize across io,
    //   - vectorize across ji,
    //   - unroll the small inner k loop to increase ILP (factor 4).
    gemm.update()
        .tile(j, i,
              jo, io,
              ji, ii,
              tile_j, tile_i,
              TailStrategy::GuardWithIf)
        // Inner-to-outer order: ji (vectorized), k (reduction),
        // then ii (rows in tile), jo (col tiles), io (row tiles).
        .reorder(ji, k, ii, jo, io)
        .parallel(io)
        .vectorize(ji, vec)
        .unroll(k, 4);

    // -------------------------------------------------------------------------
    // Bind the ImageParams to the concrete Buffers
    // -------------------------------------------------------------------------
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