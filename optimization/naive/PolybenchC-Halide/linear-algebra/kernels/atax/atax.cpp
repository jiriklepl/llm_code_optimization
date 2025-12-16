#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

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
    // Halide layout (rule 1: first dimension is fastest-varying):
    //   A(j, i) with Buffer shape (n, m)
    //   x(j)    with Buffer shape (n)

    Var i("i"), j("j");

    Func init_A("init_A"), init_x("init_x");

    // fn = (DATA_TYPE)n;
    Expr fn = cast<num_t>(Expr(n));

    // -------------------------------------------------------------------------
    // for (i = 0; i < n; i++)
    //     x[i] = 1 + (i / fn);
    // -------------------------------------------------------------------------
    {
        Expr one  = cast<num_t>(Expr(1));
        Expr j_f  = cast<num_t>(j);
        init_x(j) = one + j_f / fn;
    }

    // -------------------------------------------------------------------------
    // for (i = 0; i < m; i++)
    //   for (j = 0; j < n; j++)
    //     A[i][j] = (DATA_TYPE)((i + j) % n) / (5*m);
    //
    // With A(j, i) in Halide:
    //   A(j, i) = ((i + j) % n) / (5*m)
    // -------------------------------------------------------------------------
    {
        Expr num   = cast<num_t>((i + j) % n);
        Expr denom = Expr(5 * m);  // promoted to num_t automatically in the division
        init_A(j, i) = num / denom;
    }

    // A simple schedule for initialization. It runs once, so we keep it modest:
    // - make j the inner loop for good spatial locality in both x and A
    // - parallelize outer loops to use cores when initializing large problems
    Target target = get_host_target();
    const int vec_width = Halide::natural_vector_size<num_t>(target);

    // x is 1D; parallelize and vectorize across j
    if (vec_width > 1) {
        init_x.vectorize(j, vec_width);
    }
    init_x.parallel(j);

    // A is 2D; j is inner, i is outer
    init_A.reorder(j, i);
    if (vec_width > 1) {
        init_A.vectorize(j, vec_width);
    }
    init_A.parallel(i);

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
    // A(i,j) in C becomes A(j,i) with shape (n, m) in Halide.
    Halide::Buffer<num_t, 2> A(n, m);  // [j, i]
    Halide::Buffer<num_t, 1> x(n);     // [j]
    Halide::Buffer<num_t, 1> y(n);     // [j] output

    // Initialize data
    init_array(m, n, A, x);

    // Wrap A and x as ImageParams so the pipeline is decoupled from
    // the concrete Buffers and can be JIT-compiled independently.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");

    // Inform Halide about the memory layout to enable better
    // vectorization and bounds simplification:
    //
    //  - A(j, i): j is contiguous (unit stride)
    //  - x(j)   : contiguous
    A_param.dim(0).set_stride(1);
    x_param.dim(0).set_stride(1);

    // Vars and reduction domains
    Var i("i"), j("j");

    // Use the runtime extents of A to define the reduction ranges,
    // mirroring the original C loops over N and M.
    //
    // r1: corresponds to j in [0..n-1] for the tmp reduction
    // r2: corresponds to i in [0..m-1] for the y reduction
    RDom r1(0, A.dim(0).extent(), "r1");  // over j
    RDom r2(0, A.dim(1).extent(), "r2");  // over i

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
    // Goal:
    //   - Preserve the reduction order over r1 and r2 to match the C
    //     accumulation order for each tmp(i) and y(j).
    //   - Exploit parallelism across independent output indices (i and j).
    //   - Exploit SIMD by vectorizing across the fastest-varying
    //     dimensions (i for tmp, j for y).
    // -------------------------------------------------------------------------

    Target target = get_host_target();
    const int vec_width = Halide::natural_vector_size<num_t>(target);

    // Compute tmp once, in full, like the first nested loop in the C code.
    // For each i, r1 performs a sequential reduction over j; we parallelize
    // across i (rows) and vectorize across i as well.
    tmp.compute_root();

    tmp.parallel(i);
    if (vec_width > 1) {
        tmp.vectorize(i, vec_width);
    }

    // The update definition has the same pure var 'i', so we schedule it
    // similarly. This yields:
    //   parallel for i:
    //     tmp(i) = 0;
    //     for r1: tmp(i) += A(r1, i) * x(r1);
    tmp.update()
        .parallel(i);
    if (vec_width > 1) {
        tmp.update().vectorize(i, vec_width);
    }

    // Now schedule y(j). Each j is independent: for each j there is a
    // sequential reduction over r2 (i). We can safely parallelize and
    // vectorize across j without touching the reduction variable r2.
    y_func.compute_root();

    y_func.parallel(j);
    if (vec_width > 1) {
        y_func.vectorize(j, vec_width);
    }

    y_func.update()
        .parallel(j);
    if (vec_width > 1) {
        y_func.update().vectorize(j, vec_width);
    }

    // -------------------------------------------------------------------------
    // Bind actual Buffers to the ImageParams and JIT-compile
    // -------------------------------------------------------------------------
    A_param.set(A);
    x_param.set(x);

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