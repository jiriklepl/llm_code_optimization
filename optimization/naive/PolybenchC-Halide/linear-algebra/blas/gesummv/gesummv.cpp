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
    init_x(i) = cast<num_t>(i % n) / cast<num_t>(Expr(n));

    // A[i][j] = ((i * j + 1) % n) / n;
    // Stored as A(j, i) in the Buffer.
    init_A(j, i) = cast<num_t>((i * j + 1) % n) / cast<num_t>(Expr(n));

    // B[i][j] = ((i * j + 2) % n) / n;
    // Stored as B(j, i) in the Buffer.
    init_B(j, i) = cast<num_t>((i * j + 2) % n) / cast<num_t>(Expr(n));

    // Simple, locality-friendly schedules for initialization:
    //  - parallelize over the slow i dimension (rows)
    //  - vectorize across the fast j dimension (columns) for matrices
    init_x
        .compute_root()
        .parallel(i);

    init_A
        .compute_root()
        .parallel(i)
        .vectorize(j, 8);

    init_B
        .compute_root()
        .parallel(i)
        .vectorize(j, 8);

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

    // Halide vars and reduction domain.
    Var i("i");
    // j runs over columns; use the width of A (dim 0 == j)
    RDom j(0, A.dim(0).extent(), "j");

    // Optimized pipeline:
    //
    // Instead of computing two separate matrix-vector products
    //   tmp[i]   = sum_j A[i][j] * x[j]
    //   y_acc[i] = sum_j B[i][j] * x[j]
    // and then combining them,
    //
    // we compute both dot-products in a *single* reduction using a
    // Tuple-valued Func `dot(i) = {sumA, sumB}`. This keeps data
    // localized (single pass over A, B, x) while preserving the exact
    // accumulation order for each sum, so results are bit-identical to
    // the original two-pass formulation.

    Func dot("dot"), y_func("y_func");

    // dot(i)[0] will accumulate sum_j A[i][j] * x[j]
    // dot(i)[1] will accumulate sum_j B[i][j] * x[j]
    dot(i) = {cast<num_t>(0), cast<num_t>(0)};

    // Reduction over j: update both components of the tuple.
    Expr a_term = A_param(j, i) * x_param(j);
    Expr b_term = B_param(j, i) * x_param(j);
    dot(i) = {dot(i)[0] + a_term,
              dot(i)[1] + b_term};

    // Final combination:
    // y[i] = alpha * dotA(i) + beta * dotB(i)
    y_func(i) = Expr(alpha) * dot(i)[0] + Expr(beta) * dot(i)[1];

    // Scheduling:
    //
    // 1. Compute the final result at the root and parallelize across i
    //    (rows). Each i is independent, so this gives good
    //    multi-core scaling on a many-core CPU.
    //
    // 2. Compute the per-row reduction `dot(i)` inside the loop over i.
    //    This keeps the working set small and improves cache locality:
    //    for each row i we walk across all columns j once and then
    //    immediately consume the partial sums to form y[i].
    //
    // We deliberately avoid vectorizing across the reduction variable j
    // here to keep the reduction simple and strictly sequential per i;
    // this preserves a straightforward accumulation order, and the
    // parallelization across rows already gives substantial speedup.

    y_func
        .compute_root()
        .parallel(i);

    dot
        .compute_at(y_func, i);

    // Compile the kernel to machine code.
    y_func.compile_jit();

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