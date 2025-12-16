#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "gesummv.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize arrays A, B, x and scalars alpha, beta exactly as in the
 * original PolyBench C version.
 *
 * C layout:
 *   A[i][j], B[i][j] with i as row, j as column.
 *
 * Halide layout:
 *   Buffer<num_t,2> A(j, i)
 *   Buffer<num_t,2> B(j, i)
 *   Buffer<num_t,1> x(i)
 *
 * i is the slow (row) dimension, j is the fast (column) dimension.
 */
static
void init_array(int n,
                num_t *alpha,
                num_t *beta,
                Halide::Buffer<num_t, 2> A,
                Halide::Buffer<num_t, 2> B,
                Halide::Buffer<num_t, 1> x) {
    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j");

    Func init_A("init_A"), init_B("init_B"), init_x("init_x");

    // x[i] = (i % n) / n;
    init_x(i) = cast<num_t>(i % n) / cast<int>(n);

    // A[i][j] = ((i * j + 1) % n) / n;
    // Stored as A(j, i) in the Buffer.
    init_A(j, i) = cast<num_t>((i * j + 1) % n) / cast<int>(n);

    // B[i][j] = ((i * j + 2) % n) / n;
    // Stored as B(j, i) in the Buffer.
    init_B(j, i) = cast<num_t>((i * j + 2) % n) / cast<int>(n);

    init_x.realize(x);
    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem size.
    int n = N;

    // Scalars.
    num_t alpha;
    num_t beta;

    // Buffers:
    //  - A, B are N x N matrices, stored as (j, i) = (column, row)
    //  - x, y are N-length vectors.
    Halide::Buffer<num_t, 2> A(n, n);
    Halide::Buffer<num_t, 2> B(n, n);
    Halide::Buffer<num_t, 1> x(n);
    Halide::Buffer<num_t, 1> y(n);

    // ImageParams for A, B, x to feed into the Halide pipeline.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");

    // Initialize data.
    init_array(n, &alpha, &beta, A, B, x);

    // Bind Buffers to ImageParams.
    A_param.set(A);
    B_param.set(B);
    x_param.set(x);

    // ------------------------------------------------------------------
    // Halide algorithm: optimized GESUMMV
    //
    // We compute, for each row i:
    //   tmpA[i] = sum_j A[i,j] * x[j]
    //   tmpB[i] = sum_j B[i,j] * x[j]
    //   y[i]    = alpha * tmpA[i] + beta * tmpB[i]
    //
    // A and B are stored as A(j, i), B(j, i) so that j (column) is the
    // innermost/contiguous dimension. We fuse the two dot-products into
    // a single reduction over j using a Tuple, which:
    //   - traverses each row of A and B only once,
    //   - loads x[j] once per j and reuses it for both accumulators,
    //   - preserves the original reduction order per i and j.
    // ------------------------------------------------------------------

    Var i("i");
    RDom j(0, A.dim(0).extent(), "j");  // j runs over columns

    // accum(i) is a 2-element tuple:
    //   accum(i)[0] == tmpA[i]
    //   accum(i)[1] == tmpB[i]
    Func accum("accum"), y_func("y_func");

    // Pure definition: initialize both accumulators to zero.
    accum(i) = Tuple(cast<num_t>(0), cast<num_t>(0));

    // Reduction over j: update both accumulators atomically from the
    // previous tuple value. This is equivalent to:
    //
    //   tmpA[i] += A[i,j] * x[j];
    //   tmpB[i] += B[i,j] * x[j];
    //
    // with a single pass over j.
    accum(i) = Tuple(
        accum(i)[0] + A_param(j, i) * x_param(j),
        accum(i)[1] + B_param(j, i) * x_param(j)
    );

    // Final combination: y[i] = alpha * tmpA[i] + beta * tmpB[i].
    y_func(i) = Expr(alpha) * accum(i)[0] + Expr(beta) * accum(i)[1];

    // ------------------------------------------------------------------
    // Schedule: improve data locality, SIMD utilization and parallelism.
    //
    // - Rows i are independent -> parallelize across i in tiles.
    // - Columns j are contiguous in memory for A, B, and x ->
    //   vectorize reduction over j.
    // - Use a modest tile size in i to get good cache behavior.
    // ------------------------------------------------------------------

    // Target and natural SIMD width for num_t on this CPU.
    Target target = get_host_target();
    int vec_width = natural_vector_size(target, type_of<num_t>());
    if (vec_width < 1) {
        // Fallback: a conservative small width (should rarely be used).
        vec_width = 4;
    }

    const int tile_i = 64;  // tile size in i (rows); can be tuned.

    Var io("io"), ii("ii");

    // Compute the tuple-valued reduction at root:
    //   - Tile i into chunks of tile_i rows.
    //   - Parallelize across row tiles.
    accum.compute_root();
    accum.split(i, io, ii, tile_i)
         .parallel(io);

    // Schedule the reduction update:
    //   - Use the same tiling over i as the pure definition.
    //   - Parallelize across row tiles.
    //   - Vectorize over the reduction variable j to get SIMD FMAs.
    accum.update()
         .split(i, io, ii, tile_i)
         .parallel(io)
         .vectorize(j, vec_width);

    // y_func is a simple pointwise combination of accum; it is cheap
    // but we still parallelize and vectorize it to get contiguous writes.
    y_func.compute_root();
    y_func.split(i, io, ii, tile_i)
          .parallel(io)
          .vectorize(ii, vec_width);

    // ------------------------------------------------------------------
    // Compile the kernel to machine code.
    // ------------------------------------------------------------------
    y_func.compile_jit(target);

    // Measure execution time.
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel.
    y_func.realize(y);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Print results (mainly to prevent dead-code elimination and allow checking).
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int idx = 0; idx < n; idx++) {
            std::cerr << y(idx) << '\n';
        }
    }

    // Print timing in seconds on stdout.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}