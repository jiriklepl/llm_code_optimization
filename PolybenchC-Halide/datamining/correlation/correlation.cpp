#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (N, M, etc.)
#include "correlation.hpp"

using num_t = DATA_TYPE;

using namespace Halide;


/**
 * Initialize the input array `data` and the scalar `float_n`.
 * This mirrors the PolyBench init_array() logic:
 *
 *   *float_n = (DATA_TYPE)N;
 *   data[i][j] = (DATA_TYPE)(i*j)/M + i;
 *
 * In Halide layout, data(i,j) in C is mapped to data(j,i) in Buffer.
 */
static
void init_array(int m,
                int n,
                num_t *float_n,
                Halide::Buffer<num_t, 2> data)
{
    // float_n = (DATA_TYPE) n;
    *float_n = static_cast<num_t>(n);

    // Use Halide to fill the data buffer.
    Var i("i_init"), j("j_init");
    Func init_data("init_data");

    // C code: data[i][j] = (DATA_TYPE)(i*j)/M + i;
    // Here, i and j are ints; M is 'm'.
    // We must ensure the division is done in floating point, so we cast
    // both numerator and denominator to num_t.
    Expr ij      = i * j;
    Expr num     = cast<num_t>(ij);
    Expr denom   = cast<num_t>(Expr(m));
    Expr term_i  = cast<num_t>(i);
    init_data(j, i) = num / denom + term_i;

    // Realize into the provided buffer (width = m, height = n).
    init_data.realize(data);
}


int main(int argc, char *argv[]) {
    // Problem size (from correlation.hpp / defines.hpp).
    int n = N; // number of rows (samples)
    int m = M; // number of columns (variables)

    // Scalar corresponding to float_n in the original C code.
    num_t float_n = static_cast<num_t>(n);

    // Buffers:
    //  - data: N x M in C as data[i][j]
    //    => Buffer width = M (j), height = N (i) => data(j, i)
    //  - corr: M x M in C as corr[i][j]
    //    => Buffer width = M (j), height = M (i) => corr(j, i)
    Halide::Buffer<num_t, 2> data(m, n);
    Halide::Buffer<num_t, 2> corr_buf(m, m);

    // Initialize the input data (and float_n, though we also set float_n directly).
    init_array(m, n, &float_n, data);

    // --- Build Halide correlation pipeline ---

    // ImageParam for the input data.
    // Layout: data_param(j, i) corresponds to C data[i][j].
    ImageParam data_param(type_of<num_t>(), 2, "data_param");

    // Scalar parameter for float_n.
    Param<num_t> float_n_param("float_n_param");

    // Vars: i = row index (0..n-1), j = column index (0..m-1)
    //       p,q = indices for correlation matrix (0..m-1)
    Var i("i"), j("j"), p("p"), q("q");

    // Column-wise mean: mean[j]
    // Original C:
    //   mean[j] = 0;
    //   for (i = 0; i < N; i++)
    //       mean[j] += data[i][j];
    //   mean[j] /= float_n;
    //
    // We implement: mean[j] = sum_i data[i][j] / float_n.
    Func mean("mean");
    {
        RDom r_mean(0, n, "r_mean");
        Expr zero = cast<num_t>(Expr(0.0));

        mean(j) = zero;
        // Incremental average: sum of data / float_n
        mean(j) += data_param(j, r_mean) / float_n_param;
    }

    // Column-wise variance (without dividing by float_n yet):
    //   var[j] = sum_i (data[i][j] - mean[j])^2
    Func var("var");
    {
        RDom r_var(0, n, "r_var");
        Expr zero = cast<num_t>(Expr(0.0));

        var(j) = zero;
        Expr diff = data_param(j, r_var) - mean(j);
        var(j) += diff * diff;
    }

    // Standard deviation with epsilon handling:
    // Original C:
    //   stddev[j] = 0;
    //   for (i) stddev[j] += (data[i][j] - mean[j])^2;
    //   stddev[j] /= float_n;
    //   stddev[j] = sqrt(stddev[j]);
    //   stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    //
    // We compute:
    //   sigma = sqrt(var[j] / float_n);
    //   stddev[j] = (sigma <= eps) ? 1.0 : sigma;
    Func stddev("stddev");
    {
        Expr eps = cast<num_t>(Expr(0.1));  // SCALAR_VAL(0.1)
        Expr one = cast<num_t>(Expr(1.0));
        Expr sigma = sqrt(var(j) / float_n_param);
        stddev(j) = select(sigma <= eps, one, sigma);
    }

    // Center and reduce the column vectors:
    // Original C:
    //   for (i = 0; i < N; i++)
    //     for (j = 0; j < M; j++) {
    //       data[i][j] -= mean[j];
    //       data[i][j] /= sqrt(float_n) * stddev[j];
    //     }
    //
    // We'll compute a new Func `normalized(j, i)` instead of modifying data in place:
    //   normalized(j, i) = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
    Func normalized("normalized");
    {
        Expr sqrt_n = sqrt(float_n_param);
        normalized(j, i) = (data_param(j, i) - mean(j)) /
                           (sqrt_n * stddev(j));
    }

    // Correlation matrix:
    // Original C:
    //   for (i = 0; i < M-1; i++) {
    //     corr[i][i] = 1.0;
    //     for (j = i+1; j < M; j++) {
    //       corr[i][j] = 0.0;
    //       for (k = 0; k < N; k++)
    //         corr[i][j] += data[k][i] * data[k][j];
    //       corr[j][i] = corr[i][j];
    //     }
    //   }
    //   corr[M-1][M-1] = 1.0;
    //
    // Using normalized data, the off-diagonals are dot products of
    // normalized columns. We compute the full symmetric matrix and then
    // explicitly set the diagonal to 1.0 to match the original semantics.
    Func corr_base("corr_base");
    {
        RDom r_k(0, n, "r_k");
        Expr zero = cast<num_t>(Expr(0.0));

        corr_base(q, p) = zero;
        corr_base(q, p) += normalized(q, r_k) * normalized(p, r_k);
    }

    Func corr("corr");
    {
        Expr one = cast<num_t>(Expr(1.0));
        // corr_buf(q, p) corresponds to C corr[p][q]
        corr(q, p) = select(p == q, one, corr_base(q, p));
    }

    // Simple schedule: compute intermediate stages at root.
    // This keeps the structure close to the original C code.
    mean.compute_root();
    var.compute_root();
    stddev.compute_root();
    normalized.compute_root();

    // Bind parameters and input buffer.
    data_param.set(data);
    float_n_param.set(float_n);

    // Build a pipeline for the final correlation Func.
    Pipeline pipeline(corr);

    // JIT compile the pipeline.
    pipeline.compile_jit();

    // Time only the kernel execution (not initialization).
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: realize corr into corr_buf.
    pipeline.realize(corr_buf);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result, similar to gemm.cpp.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < m; ii++) {
            for (int jj = 0; jj < m; jj++) {
                // corr_buf(jj, ii) corresponds to corr[ii][jj] in C.
                std::cerr << corr_buf(jj, ii) << '\n';
            }
        }
    }

    // Print timing (seconds).
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}