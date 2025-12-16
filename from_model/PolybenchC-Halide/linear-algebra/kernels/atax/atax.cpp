#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "atax.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// -----------------------------------------------------------------------------
// Array initialization translated to Halide
// -----------------------------------------------------------------------------
static void init_array(int m, int n,
                       Halide::Buffer<num_t, 2> A,  // shape: [n, m]  (j, i)
                       Halide::Buffer<num_t, 1> x)  // shape: [n]     (j)
{
    // C layout:
    //   A[i][j] with i in [0..m-1], j in [0..n-1]
    //   x[j]     with j in [0..n-1]
    //
    // Halide layout (rule: first dimension is fastest-varying):
    //   A(j, i) with Buffer shape (n, m)  -- contiguous in j
    //   x(j)    with Buffer shape (n)

    Var i("i"), j("j");

    Func init_A("init_A"), init_x("init_x");

    // fn = (DATA_TYPE)n;
    Expr fn = cast<num_t>(Expr(n));

    // for (j = 0; j < n; j++)
    //     x[j] = 1 + (j / fn);
    {
        Expr one = cast<num_t>(Expr(1));
        Expr j_f = cast<num_t>(j);
        init_x(j) = one + j_f / fn;
    }

    // for (i = 0; i < m; i++)
    //   for (j = 0; j < n; j++)
    //     A[i][j] = (DATA_TYPE)((i + j) % n) / (5*m);
    {
        Expr num = cast<num_t>((i + j) % n);
        Expr denom = Expr(5 * m);  // promoted to num_t automatically
        // In the Halide buffer we store this as A(j, i) with shape (n, m).
        init_A(j, i) = num / denom;
    }

    // Realize into the provided Buffers
    init_x.realize(x);
    init_A.realize(A);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Problem size (from atax.hpp / defines.hpp)
    int m = M;
    int n = N;

    // Buffers:
    // A(i,j) in C becomes A(j,i) with shape (n, m) in Halide so that the
    // column index j is the innermost/contiguous dimension.
    Halide::Buffer<num_t, 2> A(n, m);  // [j, i]
    Halide::Buffer<num_t, 1> x(n);     // [j]
    Halide::Buffer<num_t, 1> y(n);     // [j] output

    // Initialize data
    init_array(m, n, A, x);

    // Wrap A and x as ImageParams so the pipeline is decoupled from
    // the concrete Buffers and can be JIT-compiled independently.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");

    // Vars and reduction domains
    Var i("i"), j("j");
    // r1 corresponds to the original column loop j in [0..n-1]
    // r2 corresponds to the original row loop   i in [0..m-1]
    RDom r1(0, A.dim(0).extent(), "r1");  // j
    RDom r2(0, A.dim(1).extent(), "r2");  // i

    // -------------------------------------------------------------------------
    // Kernel translation
    //
    // Original C kernel:
    //
    //   for (i = 0; i < _PB_N; i++)
    //     y[i] = 0;
    //   for (i = 0; i < _PB_M; i++) {
    //     tmp[i] = 0;
    //     for (j = 0; j < _PB_N; j++)
    //       tmp[i] = tmp[i] + A[i][j] * x[j];
    //     for (j = 0; j < _PB_N; j++)
    //       y[j] = y[j] + A[i][j] * tmp[i];
    //   }
    //
    // This is:
    //   tmp[i] = sum_j A[i][j] * x[j];
    //   y[j]   = sum_i A[i][j] * tmp[i];
    //
    // With our layout A(j,i):
    //   tmp(i) = sum_j A(j,i) * x(j);
    //   y(j)   = sum_i A(j,i) * tmp(i);
    // -------------------------------------------------------------------------
    Func tmp("tmp"), y_func("y_func");

    // tmp(i) = sum over r1 (j) of A(j,i) * x(j)
    tmp(i) = cast<num_t>(Expr(0));
    tmp(i) += A_param(r1, i) * x_param(r1);

    // y(j) = sum over r2 (i) of A(j,i) * tmp(i)
    y_func(j) = cast<num_t>(Expr(0));
    y_func(j) += A_param(j, r2) * tmp(r2);

    // -------------------------------------------------------------------------
    // Schedule
    //
    // We follow the intent of the atax-* models:
    //   - Stage 1: tmp = A * x
    //       * parallelize across rows i (independent dot-products)
    //       * use contiguous j (r1) as the innermost/vectorized loop
    //   - Stage 2: y = A^T * tmp
    //       * parallelize across output indices j
    //       * vectorize across j (contiguous in memory)
    //       * keep r2 (i) as a serial reduction outside the vector lane
    //
    // This preserves the mathematical computation, but significantly
    // improves data locality and SIMD utilization on a multicore x64 CPU.
    // -------------------------------------------------------------------------

    // Choose a reasonable vector width for the current target and data type.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());
    // Tile size along j for the final y reduction. This is a coarse-grain
    // parallelism granularity; 256 is a good default for large N.
    const int tile_j = 256;

    // -------------------------
    // Stage 1: tmp = A * x
    // -------------------------
    //
    // Each tmp(i) is an independent dot-product:
    //   tmp(i) = sum_{r1} A(r1, i) * x(r1)
    //
    // We:
    //   - compute tmp at root so it is fully materialized before y
    //   - parallelize across i (rows of A / entries of tmp)
    //   - reorder the reduction so that r1 (j) is innermost
    //   - vectorize across r1 to stream contiguous A(j,i) and x(j)
    //
    tmp.compute_root();

    // Parallelize across rows i: each i is an independent reduction.
    tmp.parallel(i);

    // Reorder update loops to iterate r1 (j) innermost for better locality,
    // then vectorize r1 over the natural vector width.
    tmp.update()
        .reorder(r1, i)
        .vectorize(r1, vec_width);

    // -------------------------
    // Stage 2: y = A^T * tmp
    // -------------------------
    //
    //   y(j) = sum_{r2} A(j, r2) * tmp(r2)
    //
    // We want:
    //   - j-contiguous traversal for A(j,r2) and y(j)
    //   - parallelism across disjoint ranges of j
    //   - vectorization across j for SIMD updates
    //
    // We achieve this by tiling j into [jo, ji], parallelizing jo,
    // and vectorizing ji, both in the pure and update stages.
    //
    y_func.compute_root();

    Var jo("jo"), ji("ji");

    // Pure definition: y(j) = 0;
    // Tile and parallelize across j, vectorize the inner tile.
    y_func
        .split(j, jo, ji, tile_j)
        .parallel(jo)
        .vectorize(ji, vec_width);

    // Update definition: y(j) += A(j,r2) * tmp(r2);
    // Make j-tile indices explicit, keep r2 as a serial reduction
    // outside the innermost vectorized j loop.
    y_func.update()
        .split(j, jo, ji, tile_j)
        // Loop order: outer tiles jo, then reduction r2, innermost ji.
        // Any order of jo and r2 is legal because additions over r2 are
        // associative; this order gives good cache behavior:
        //   for jo (parallel):
        //     for r2:
        //       for ji (vectorized):
        //         y[...] += A(...) * tmp(r2)
        .reorder(ji, r2, jo)
        .parallel(jo)
        .vectorize(ji, vec_width);

    // Bind actual Buffers to the ImageParams
    A_param.set(A);
    x_param.set(x);

    // JIT-compile the kernel for the chosen target
    y_func.compile_jit(target);

    // -------------------------------------------------------------------------
    // Run and time the kernel
    // -------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: realize y_func into y buffer
    y_func.realize(y);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -------------------------------------------------------------------------
    // Print results (to prevent dead-code elimination and for checking)
    // -------------------------------------------------------------------------
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int idx = 0; idx < n; idx++) {
            std::cerr << y(idx) << '\n';
        }
    }

    // Print execution time in seconds (like gemm.cpp)
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}