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
static void init_array(int n,
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
    init_A(j, i) = cast<num_t>((i * j) % n) / cast<int>(n);

    // ---------------------------
    // Scheduling for initialization
    // ---------------------------
    // Use a reasonable native vector width for the host and parallelize
    // across the outer dimension where appropriate.
    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Initialize 1D vectors in parallel, vectorizing across i.
    init_x1.compute_root()
        .vectorize(i, vec_width)
        .parallel(i);
    init_x2.compute_root()
        .vectorize(i, vec_width)
        .parallel(i);
    init_y1.compute_root()
        .vectorize(i, vec_width)
        .parallel(i);
    init_y2.compute_root()
        .vectorize(i, vec_width)
        .parallel(i);

    // Initialize matrix A row by row (i is the C row index, dim 1 here),
    // vectorizing across j (dim 0, contiguous in memory).
    init_A.compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

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
    //
    // We implement both updates in a single Tuple-valued Func so that we:
    //   * Traverse i only once.
    //   * Share the reduction loop over j (r) for both x1 and x2.
    //   * Allow the compiler to fuse work and improve cache locality for
    //     accesses to y_1 / y_2 and loop overhead.
    Var i("i");
    // Use the width of A (dim 0, corresponding to j) as the reduction extent.
    RDom r(0, A.dim(0).extent(), "r");

    Func mvt("mvt");

    // Pure definition: start from the initial x1 and x2.
    // mvt(i)[0] will hold the running value for x1[i],
    // mvt(i)[1] will hold the running value for x2[i].
    mvt(i) = Tuple(x1_param(i), x2_param(i));

    // Update definition: accumulate the contributions over r (the j loop).
    // We read the current values from the Tuple and write back updated ones.
    Expr x1_old = mvt(i)[0];
    Expr x2_old = mvt(i)[1];

    Expr x1_new = x1_old + A_param(r, i) * y1_param(r);  // A[i][j] * y_1[j]
    Expr x2_new = x2_old + A_param(i, r) * y2_param(r);  // A[j][i] * y_2[j]

    mvt(i) = Tuple(x1_new, x2_new);

    // ---------------------------
    // Scheduling for the MVT kernel
    // ---------------------------
    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Tile the i dimension to expose coarse-grain parallelism and fine-grain
    // vectorization; this works for both the pure and the update definitions.
    Var io("io"), ii("ii");

    const int tile_i = 64;  // outer tile size along i; chosen as a reasonable default

    mvt.compute_root()
        .split(i, io, ii, tile_i)
        .parallel(io)          // parallelize tiles across CPU cores
        .vectorize(ii, vec_width); // vectorize within each tile

    // Apply the same parallel/vector scheme to the reduction update step so
    // that the (i, r) nest benefits from the same layout.
    mvt.update()
        .parallel(io)
        .vectorize(ii, vec_width);

    // Explicitly JIT-compile before timing so compilation time is not included.
    mvt.compile_jit(target);

    // Time the kernel execution (both x1 and x2 updates)
    auto start = std::chrono::high_resolution_clock::now();

    // Realize both updated vectors back into x1 and x2.
    //
    // We create a Realization that wraps the existing buffers so that the
    // Func writes directly into them; this preserves the "in-place" view
    // of x1/x2 from the original C code.
    {
        Realization out({x1, x2});
        mvt.realize(out);
    }

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