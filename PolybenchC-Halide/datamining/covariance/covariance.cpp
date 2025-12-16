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
    // Simple schedule
    // -----------------------------

    // Compute the intermediate reductions at root, in separate passes,
    // roughly mirroring the phase structure of the original C code.
    mean_sum.compute_root();
    mean.compute_root();
    cov_sum.compute_root();
    covariance.compute_root();

    // Bind runtime parameters.
    data_param.set(data);
    float_n_param.set(float_n);

    // JIT-compile the final stage.
    covariance.compile_jit();

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