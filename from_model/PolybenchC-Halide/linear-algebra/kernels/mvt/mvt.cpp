#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (defines DATA_TYPE, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "mvt.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Array initialization translated to Halide.
 *
 * Original C:
 *   for (i = 0; i < n; i++) {
 *     x1[i]  = (DATA_TYPE) (i % n) / n;
 *     x2[i]  = (DATA_TYPE) ((i + 1) % n) / n;
 *     y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
 *     y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
 *     for (j = 0; j < n; j++)
 *       A[i][j] = (DATA_TYPE) (i*j % n) / n;
 *   }
 */
static
void init_array(int n,
                Halide::Buffer<num_t, 1> x1,
                Halide::Buffer<num_t, 1> x2,
                Halide::Buffer<num_t, 1> y_1,
                Halide::Buffer<num_t, 1> y_2,
                Halide::Buffer<num_t, 2> A) {
    Var i("i"), j("j");

    Func init_x1("init_x1");
    Func init_x2("init_x2");
    Func init_y1("init_y1");
    Func init_y2("init_y2");
    Func init_A("init_A");

    // 1D vectors
    init_x1(i) = cast<num_t>(i % n) / cast<int>(n);
    init_x2(i) = cast<num_t>((i + 1) % n) / cast<int>(n);
    init_y1(i) = cast<num_t>((i + 3) % n) / cast<int>(n);
    init_y2(i) = cast<num_t>((i + 4) % n) / cast<int>(n);

    // 2D matrix A: C layout A[i][j] -> Halide A(j, i)
    // j is dim 0 and contiguous in memory; i is dim 1.
    init_A(j, i) = cast<num_t>((i * j) % n) / cast<int>(n);

    // Realize into the provided buffers
    init_x1.realize(x1);
    init_x2.realize(x2);
    init_y1.realize(y_1);
    init_y2.realize(y_2);
    init_A.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Buffers corresponding to:
    //   DATA_TYPE A[N][N];
    //   DATA_TYPE x1[N], x2[N], y_1[N], y_2[N];
    //
    // PolyBench uses A[i][j]; we map j -> dim 0 (x), i -> dim 1 (y).
    // This makes the C column index j the unit-stride dimension.
    Halide::Buffer<num_t, 2> A(n, n);     // (j, i)
    Halide::Buffer<num_t, 1> x1(n);
    Halide::Buffer<num_t, 1> x2(n);
    Halide::Buffer<num_t, 1> y_1(n);
    Halide::Buffer<num_t, 1> y_2(n);

    // ImageParams to feed into the Halide pipeline
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam x1_param(type_of<num_t>(), 1, "x1_param");
    ImageParam x2_param(type_of<num_t>(), 1, "x2_param");
    ImageParam y1_param(type_of<num_t>(), 1, "y1_param");
    ImageParam y2_param(type_of<num_t>(), 1, "y2_param");

    // Initialize data using a Halide-based initializer
    init_array(n, x1, x2, y_1, y_2, A);

    // Bind buffers to ImageParams
    A_param.set(A);
    x1_param.set(x1);
    x2_param.set(x2);
    y1_param.set(y_1);
    y2_param.set(y_2);

    // Halide translation of kernel_mvt:
    //
    // Original C kernel:
    //
    // for (i = 0; i < N; i++)
    //   for (j = 0; j < N; j++)
    //     x1[i] = x1[i] + A[i][j] * y_1[j];
    // for (i = 0; i < N; i++)
    //   for (j = 0; j < N; j++)
    //     x2[i] = x2[i] + A[j][i] * y_2[j];
    //
    // Mathematically:
    //   x1[i] = x1[i] + sum_j A[i][j] * y_1[j]
    //   x2[i] = x2[i] + sum_j A[j][i] * y_2[j]
    //
    // With our (j, i) mapping for A[i][j]:
    //   A[i][j]  -> A_param(j, i)
    //   A[j][i]  -> A_param(i, j)

    Var i("i");
    // Use the width of A (dim 0, corresponding to j) as the reduction extent.
    RDom r(0, A.dim(0).extent(), "r");

    Func mvt_x1("mvt_x1");
    Func mvt_x2("mvt_x2");

    // First loop nest: accumulate into x1
    // x1[i] = x1[i] + sum_j A[i][j] * y_1[j]
    // In Halide coordinates: A[i][j] == A_param(j, i), so A_param(r, i)
    mvt_x1(i) = x1_param(i);
    mvt_x1(i) += A_param(r, i) * y1_param(r);

    // Second loop nest: accumulate into x2
    // x2[i] = x2[i] + sum_j A[j][i] * y_2[j]
    // A[j][i] == A_param(i, j), so A_param(i, r)
    mvt_x2(i) = x2_param(i);
    mvt_x2(i) += A_param(i, r) * y2_param(r);

    // ----------------------------
    // Scheduling for performance
    // ----------------------------

    // Query the JIT target so we can choose a reasonable vector width.
    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // ---- Schedule for mvt_x1 ----
    //
    // Access pattern in the update:
    //   A_param(r, i) and y1_param(r)
    // For a fixed i, r iterates over the unit-stride dimension of A and y1,
    // so we want r as the innermost loop and we vectorize across r.
    //
    // Loop structure after this schedule:
    //   for i in [0, n) (parallel)
    //     for r in [0, n) (vectorized)
    //       x1[i] += A[r, i] * y1[r];

    mvt_x1.compute_root();

    mvt_x1.update()
          // Arguments of reorder are listed from innermost to outermost.
          // This makes r inner, i outer: for (i) { for (r) { ... } }.
          .reorder(r, i)
          // Vectorize across r: contiguous loads from A(r, i) and y1(r).
          .vectorize(r, vec_width)
          // Parallelize across rows i; each i updates a distinct x1[i].
          .parallel(i);

    // ---- Schedule for mvt_x2 ----
    //
    // Access pattern in the update:
    //   A_param(i, r) and y2_param(r)
    // A_param is row-major in its first dimension (the Halide x coordinate).
    // For good locality we want i to be the innermost loop so that we walk
    // contiguous elements A(i, r) for a fixed r (i.e., a row of the C matrix).
    //
    // We also tile i to create coarse-grain chunks for parallelism and
    // vectorize within each tile.
    //
    // Target loop structure:
    //   for io in tiles of i (parallel)
    //     for r in [0, n)
    //       t = y2[r]
    //       for ii in [0, tile_size) (vectorized)
    //         int i = io * tile_size + ii;
    //         x2[i] += A[i, r] * t;

    Var io("io"), ii("ii");
    const int tile_i = 128;  // tile size in i; should be >= vec_width

    mvt_x2.compute_root();

    mvt_x2.update()
          // Split i into outer tile index io and inner index ii.
          .split(i, io, ii, tile_i)
          // Make ii innermost (for vectorization), then r, then io outermost:
          //   for io:
          //     for r:
          //       for ii:
          //         ...
          .reorder(ii, r, io)
          // Vectorize across contiguous ii (i within a tile).
          .vectorize(ii, vec_width)
          // Parallelize across tiles of i. Each tile writes a disjoint
          // portion of x2, so there are no races.
          .parallel(io);

    // JIT-compile the two Funcs for the chosen target
    mvt_x1.compile_jit(target);
    mvt_x2.compile_jit(target);

    // Time the kernel execution (both x1 and x2 updates)
    auto start = std::chrono::high_resolution_clock::now();

    // Realize back into x1 and x2 buffers (in-place relative to the C view)
    mvt_x1.realize(x1);
    mvt_x2.realize(x2);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optional: print results to stderr (similar spirit to PolyBench print_array)
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);

        // Print x1
        for (int idx = 0; idx < n; idx++) {
            std::cerr << x1(idx) << '\n';
        }
        // Print x2
        for (int idx = 0; idx < n; idx++) {
            std::cerr << x2(idx) << '\n';
        }
    }

    // Print timing (seconds) to stdout
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}