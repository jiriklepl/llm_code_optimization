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

    // data has shape (m, n): width = M (columns), height = N (rows).
    init_data(j, i) = cast<num_t>(i * j) / cast<num_t>(m);

    // Simple but reasonably efficient schedule for initialization:
    //  - parallelize over rows i (samples)
    //  - vectorize across columns j (features), which are contiguous in memory
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    init_data
        .parallel(i)
        .vectorize(j, vec_width);

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
    // data[i][j] -> data_param(j, i)
    ImageParam data_param(type_of<num_t>(), 2, "data_param");

    // Scalar parameter float_n.
    Param<num_t> float_n_param("float_n_param");

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

    // 2. Center the data by subtracting the mean from each column:
    //
    // C code:
    // for (i = 0; i < N; i++)
    //   for (j = 0; j < M; j++)
    //     data[i][j] -= mean[j];
    //
    // We define a new Func 'centered' instead of modifying data in-place.
    // Signature: centered(j, i) = data_param(j, i) - mean(j)
    Func centered("centered");
    centered(j, i) = data_param(j, i) - mean(j);

    // 3. Compute the covariance matrix:
    //
    // C code:
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
    // cov(i,j) = sum_k centered(i,k) * centered(j,k) / (float_n - 1)
    Func cov_sum("cov_sum"), covariance("covariance");
    RDom r_samples(0, data_param.dim(1).extent(), "r_samples");  // k = 0..n-1

    cov_sum(i, j) = cast<num_t>(0);
    cov_sum(i, j) += centered(i, r_samples) * centered(j, r_samples);
    covariance(i, j) = cov_sum(i, j) / (float_n_param - cast<num_t>(1));

    // -----------------------------
    // Optimized schedule
    // -----------------------------

    // We target the host CPU, using its natural vector width for num_t.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // 1) Column means:
    //
    // mean_sum(j) is an independent reduction for each column j.
    // We:
    //  - compute it at root (once for all),
    //  - parallelize across columns j,
    //  - vectorize across j inside the reduction update so
    //    that we load contiguous elements along x (data columns).
    mean_sum.compute_root();
    mean_sum
        .update()
        .parallel(j)
        .vectorize(j, vec_width);

    // mean(j) is a simple pointwise transform of mean_sum(j).
    // Compute it once, in parallel and vectorized over j.
    mean
        .compute_root()
        .parallel(j)
        .vectorize(j, vec_width);

    // 2) Centered data:
    //
    // We do not schedule 'centered' explicitly; it is inlined into
    // the covariance reduction below. Each evaluation of centered(i,k)
    // simply evaluates data_param(i,k) - mean(i) on the fly. This keeps
    // us from materializing the full N×M centered matrix.

    // 3) Covariance:
    //
    // cov_sum(i,j) is a 2D reduction over r_samples (rows).
    // The computational cost is dominated by this stage, so we:
    //  - compute cov_sum at root,
    //  - parallelize over the outer column j (M is large enough),
    //  - vectorize over i, which is the innermost coordinate and
    //    contiguous in memory (x-dimension of the Buffer).
    cov_sum.compute_root();
    cov_sum
        .update()
        .parallel(j)          // distribute columns across threads
        .vectorize(i, vec_width); // vectorize across contiguous feature index i

    // Final scaling from cov_sum to covariance is cheap and
    // embarrassingly parallel; we just parallelize and vectorize it.
    covariance
        .compute_root()
        .parallel(j)
        .vectorize(i, vec_width);

    // -----------------------------
    // Bind runtime parameters & JIT
    // -----------------------------

    data_param.set(data);
    float_n_param.set(float_n);

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

    // As in the GEMM example, print the output to prevent dead-code elimination,
    // and provide a basic correctness check if desired.
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