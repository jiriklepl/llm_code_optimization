#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common PolyBench-style definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (N, M, etc.)
#include "covariance.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// Initialize the input array, matching the original C code semantics.
static void init_array(int m, int n,
                       num_t *float_n,
                       Halide::Buffer<num_t, 2> data) {
    // float_n = (DATA_TYPE)n;
    *float_n = (num_t)n;

    // C code:
    // for (i = 0; i < N; i++)
    //   for (j = 0; j < M; j++)
    //     data[i][j] = ((DATA_TYPE) i*j) / M;
    //
    // C array data[i][j] is mapped to Buffer data(j, i):
    Var i("i"), j("j");
    Func init_data("init_data");

    init_data(j, i) = cast<num_t>(i * j) / cast<num_t>(m);

    // Simple but cache‑friendly schedule for initialization:
    //  - Vectorize along j (contiguous in memory),
    //  - Parallelize along i (rows).
    Target target = get_jit_target_from_environment();
    const int vec_width = std::max(1, target.natural_vector_size(type_of<num_t>()));

    init_data
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // data has shape (m, n): width = M (columns), height = N (rows).
    init_data.realize(data);
}

int main(int argc, char *argv[]) {
    // Problem size (from covariance.hpp / defines.hpp)
    int n = N;
    int m = M;

    // Scalar and arrays.
    num_t float_n;

    // C: DATA_TYPE data[N][M];
    // Map to Halide: Buffer<num_t,2> data(M, N);  // (x = column j, y = row i)
    Halide::Buffer<num_t, 2> data(m, n);

    // C: DATA_TYPE cov[M][M];
    // Map to Halide: Buffer<num_t,2> cov(M, M);
    Halide::Buffer<num_t, 2> cov(m, m);

    // Initialize data and float_n.
    init_array(m, n, &float_n, data);

    // -----------------------------
    // Build the Halide pipeline
    // -----------------------------

    // ImageParam for the input data matrix.
    // data[i][j] in C  ->  data_param(j, i) here.
    ImageParam data_param(type_of<num_t>(), 2, "data_param");

    // Scalar parameter float_n.
    Param<num_t> float_n_param("float_n_param");

    // Vars:
    //  - i, j index features (columns) for mean / covariance.
    //  - r_rows, r_samples index samples (rows) as reduction domains.
    Var i("i"), j("j");

    // 1. Compute column means:
    //
    // C code:
    // for (j = 0; j < M; j++) {
    //   mean[j] = 0;
    //   for (i = 0; i < N; i++)
    //     mean[j] += data[i][j];
    //   mean[j] /= float_n;
    // }
    //
    // data[i][j] -> data_param(j, i)
    Func mean_sum("mean_sum"), mean("mean");
    RDom r_rows(0, data_param.dim(1).extent(), "r_rows");  // iterate over rows i = 0..n-1

    mean_sum(j) = cast<num_t>(0);
    mean_sum(j) += data_param(j, r_rows);   // sum over rows
    mean(j) = mean_sum(j) / float_n_param;  // divide by float_n

    // 2. Compute the covariance matrix directly from centered data
    //    without materializing a separate "centered" buffer:
    //
    // C code after centering:
    // for (i = 0; i < M; i++)
    //   for (j = i; j < M; j++) {
    //     cov[i][j] = 0;
    //     for (k = 0; k < N; k++)
    //       cov[i][j] += data[k][i] * data[k][j];
    //     cov[i][j] /= (float_n - 1.0);
    //     cov[j][i] = cov[i][j];
    //   }
    //
    // Here we compute the full MxM covariance as:
    //   cov(i,j) = sum_k (data[k][i] - mean[i]) * (data[k][j] - mean[j])
    //              / (float_n - 1)
    //
    // Using a separate reduction Func cov_sum gives us flexible
    // scheduling of the O(N*M^2) dominant phase.
    Func cov_sum("cov_sum"), covariance("covariance");
    RDom r_samples(0, data_param.dim(1).extent(), "r_samples");  // k = 0..n-1

    cov_sum(i, j) = cast<num_t>(0);
    cov_sum(i, j) += (data_param(i, r_samples) - mean(i)) *
                     (data_param(j, r_samples) - mean(j));
    covariance(i, j) = cov_sum(i, j) / (float_n_param - cast<num_t>(1));

    // -----------------------------
    // Optimized schedule
    // -----------------------------

    Target target = get_jit_target_from_environment();
    const int vec_width = std::max(1, target.natural_vector_size(type_of<num_t>()));

    // Tile sizes for feature dimensions in the covariance computation.
    const int tile_i = 32;
    const int tile_j = 32;

    // 1) Mean computation:
    //
    // mean_sum(j) is an independent reduction over rows. We:
    //  - compute it at root (once),
    //  - parallelize across column tiles,
    //  - vectorize within each tile.
    mean_sum.compute_root();
    mean.compute_root();

    {
        Var jo("jo"), ji("ji");
        RVar ri = r_rows.x;

        // Initialize mean_sum(j) = 0; default schedule is fine (O(M)).
        // Optimize the reduction update:
        mean_sum.update()
            .split(j, jo, ji, 64)          // tiles of 64 columns
            // Loop order (outermost to innermost):
            //   jo (tile of columns, parallelized)
            //   ri (rows / samples)
            //   ji (columns within the tile, vectorized)
            .reorder(ji, ri, jo)
            .parallel(jo)
            .vectorize(ji, vec_width);
    }

    {
        // mean(j) = mean_sum(j) / float_n; cheap, but we still enable
        // some parallelism and vectorization across j.
        Var jo("jo"), ji("ji");
        mean
            .compute_root()
            .split(j, jo, ji, 64)
            .parallel(jo)
            .vectorize(ji, vec_width);
    }

    // 2) Covariance computation:
    //
    // cov_sum(i,j) = sum_k (...) is O(N*M^2) and dominates runtime.
    // We treat it like a matrix-multiply style reduction and:
    //  - tile the feature dimensions (i,j),
    //  - parallelize across row tiles (io),
    //  - vectorize across the fast-varying j dimension within each tile,
    //  - keep the reduction over samples r_samples sequential to avoid
    //    race conditions.
    cov_sum.compute_root();
    covariance.compute_root();

    {
        Var io("io"), jo("jo"), ii("ii"), jj("jj");
        RVar rk = r_samples.x;

        // Pure definition (initialization to zero) – tile and parallelize.
        cov_sum
            .tile(i, j, io, jo, ii, jj, tile_i, tile_j)
            // Loop order (outermost to innermost):
            //   io (row tiles, parallelized)
            //   jo (column tiles)
            //   ii (rows within tile)
            //   jj (cols within tile, vectorized)
            .reorder(jj, ii, jo, io)
            .parallel(io)
            .vectorize(jj, vec_width);

        // Reduction update: accumulate the outer products of centered rows.
        cov_sum.update()
            .tile(i, j, io, jo, ii, jj, tile_i, tile_j)
            // New loop order (outermost to innermost):
            //   io (row tiles, parallelized)
            //   jo (column tiles)
            //   rk (samples / rows)
            //   ii (rows within tile)
            //   jj (cols within tile, vectorized)
            //
            // Passing arguments from innermost to outermost to reorder():
            //   jj (innermost), ii, rk, jo, io (outermost).
            .reorder(jj, ii, rk, jo, io)
            .parallel(io)
            .vectorize(jj, vec_width);
    }

    {
        // Final scaling of cov_sum to obtain covariance.
        // This is O(M^2), much cheaper than the reduction itself, but
        // we still tile, parallelize, and vectorize for good measure.
        Var io("io"), jo("jo"), ii("ii"), jj("jj");
        covariance
            .compute_root()
            .tile(i, j, io, jo, ii, jj, tile_i, tile_j)
            .reorder(jj, ii, jo, io)
            .parallel(io)
            .vectorize(jj, vec_width);
    }

    // Bind runtime parameters.
    data_param.set(data);
    float_n_param.set(float_n);

    // JIT-compile the final stage with the chosen target.
    covariance.compile_jit(target);

    // -----------------------------
    // Run and time the kernel
    // -----------------------------

    auto start = std::chrono::high_resolution_clock::now();

    // Realize the covariance matrix into 'cov'.
    covariance.realize(cov);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -----------------------------
    // Print results (optional)
    // -----------------------------

    using namespace std::string_literals;

    // Print the output to prevent dead-code elimination and to allow
    // simple correctness checks. We preserve the original printing
    // order and index mapping: C cov[ii][jj] -> Halide cov(jj, ii).
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < m; ii++) {
            for (int jj = 0; jj < m; jj++) {
                // C cov[ii][jj] -> Halide cov(jj, ii)
                std::cerr << cov(jj, ii) << '\n';
            }
        }
    }

    // Print timing in seconds.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}