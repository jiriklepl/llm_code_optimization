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

    // Simple schedule for initialization: parallelize rows and vectorize
    // across columns (which are unit-stride in memory).
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    init_data
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

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
    // We implement: mean[j] = sum_i data[i][j] / float_n, which is
    // algebraically equivalent to the original loop.
    Func mean("mean");
    RDom r_mean(0, n, "r_mean");
    {
        Expr zero = cast<num_t>(Expr(0.0));
        mean(j) = zero;
        mean(j) += data_param(j, r_mean) / float_n_param;
    }

    // Column-wise variance accumulator (without dividing by float_n yet):
    //   var[j] = sum_i (data[i][j] - mean[j])^2
    Func var("var");
    RDom r_var(0, n, "r_var");
    {
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

    // Precompute per-column scaling factor:
    //
    //   normalized(i,j) = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
    //                   = (data[i][j] - mean[j]) * gamma[j],
    //
    // where gamma[j] = 1 / (sqrt(float_n) * stddev[j]).
    // We use gamma so we don't need to materialize the normalized matrix.
    Func gamma("gamma");
    {
        Expr one = cast<num_t>(Expr(1.0));
        Expr sqrt_n = sqrt(float_n_param);
        gamma(j) = one / (sqrt_n * stddev(j));
    }

    // Correlation matrix:
    //
    // Using normalized data, the off-diagonals are dot products of
    // normalized columns:
    //
    //   corr_base[p,q] = sum_k normalized[k,p] * normalized[k,q].
    //
    // We expand normalized and fold in gamma:
    //
    //   normalized[k, p] = (data[k][p] - mean[p]) * gamma[p]
    //   normalized[k, q] = (data[k][q] - mean[q]) * gamma[q]
    //
    //   corr_base[p,q] = sum_k (data[k][p] - mean[p]) * gamma[p] *
    //                         (data[k][q] - mean[q]) * gamma[q]
    //
    // We still explicitly set the diagonal of corr to 1.0 afterwards
    // to match the original C semantics.
    Func corr_base("corr_base");
    RDom r_k(0, n, "r_k");
    {
        Expr zero = cast<num_t>(Expr(0.0));
        corr_base(q, p) = zero;

        Expr diff_q = data_param(q, r_k) - mean(q);
        Expr diff_p = data_param(p, r_k) - mean(p);

        corr_base(q, p) += diff_q * gamma(q) * diff_p * gamma(p);
    }

    Func corr("corr");
    {
        Expr one = cast<num_t>(Expr(1.0));
        // corr_buf(q, p) corresponds to C corr[p][q]
        corr(q, p) = select(p == q, one, corr_base(q, p));
    }

    // -------------------------
    // Scheduling for performance
    // -------------------------

    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Statistics stages (mean, var, stddev, gamma)
    //
    // We reorder the reductions to iterate over rows (i) outer and
    // columns (j) inner, which matches the row-major layout
    // data(j,i) with j contiguous. This improves data locality when
    // streaming through rows.
    {
        Var j_outer("j_outer"), j_inner("j_inner");

        // mean[j]: reduction over i
        mean.compute_root();
        mean.update()
            .split(j, j_outer, j_inner, 64)   // tile columns
            .reorder(j_inner, j_outer, r_mean) // for (r_mean) { for (j_outer) { for (j_inner) ... } }
            .parallel(j_outer)                // parallelize across column tiles
            .vectorize(j_inner, vec_width);   // SIMD across contiguous columns

        // var[j]: reduction over i
        var.compute_root();
        var.update()
            .split(j, j_outer, j_inner, 64)
            .reorder(j_inner, j_outer, r_var)
            .parallel(j_outer)
            .vectorize(j_inner, vec_width);

        // stddev[j]: pointwise over j
        stddev
            .compute_root()
            .split(j, j_outer, j_inner, 64)
            .parallel(j_outer)
            .vectorize(j_inner, vec_width);

        // gamma[j]: pointwise over j, depends only on stddev
        gamma
            .compute_root()
            .split(j, j_outer, j_inner, 64)
            .parallel(j_outer)
            .vectorize(j_inner, vec_width);
    }

    // Heavy stage: corr_base(q,p) = dot product over k.
    //
    // We treat this as a blocked GEMM-like kernel on a symmetric
    // Gram matrix. Tile p and q (variables) so that tiles of the
    // correlation matrix fit in cache, and tile k (samples) so that
    // working sets of input rows remain in L1/L2.
    {
        Var q_outer("q_outer"), p_outer("p_outer");
        Var q_inner("q_inner"), p_inner("p_inner");
        Var tile_index("tile_index");
        RVar k_outer("k_outer"), k_inner("k_inner");

        const int tile_q = 32;  // tile size along q (fast dimension)
        const int tile_p = 32;  // tile size along p
        const int tile_k = 64;  // tile size along reduction dim k

        // Pure definition: just to establish a tiled layout for corr_base.
        corr_base
            .compute_root()
            .tile(q, p, q_outer, p_outer, q_inner, p_inner, tile_q, tile_p)
            .fuse(q_outer, p_outer, tile_index)
            .parallel(tile_index)
            .vectorize(q_inner, vec_width);

        // Update definition (the reduction over k).
        corr_base.update()
            // Tile p and q in the same way as the pure definition.
            .tile(q, p, q_outer, p_outer, q_inner, p_inner, tile_q, tile_p)
            .fuse(q_outer, p_outer, tile_index)
            // Tile the reduction over k to improve cache locality.
            .split(r_k, k_outer, k_inner, tile_k)
            // Loop order: outermost k_outer, then tile_index, then
            // k_inner, p_inner, q_inner (q_inner innermost for SIMD).
            .reorder(q_inner, p_inner, k_inner, tile_index, k_outer)
            .parallel(tile_index)
            .vectorize(q_inner, vec_width);
    }

    // Final corr(q,p): cheap element-wise selection between 1.0 on the
    // diagonal and corr_base elsewhere. Parallelize rows (p) and
    // vectorize across columns (q).
    {
        Var p_outer("p_outer"), p_inner("p_inner");
        corr
            .compute_root()
            .split(p, p_outer, p_inner, 32)
            .parallel(p_outer)
            .vectorize(q, vec_width);
    }

    // Bind parameters and input buffer.
    data_param.set(data);
    float_n_param.set(float_n);

    // Build a pipeline for the final correlation Func.
    Pipeline pipeline(corr);

    // JIT compile the pipeline for the host.
    pipeline.compile_jit(target);

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