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
    // Halide layout (rule 1: first dimension is fastest-varying):
    //   A(j, i) with Buffer shape (n, m)
    //   x(j)    with Buffer shape (n)

    Var i("i"), j("j");

    Func init_A("init_A"), init_x("init_x");

    // fn = (DATA_TYPE)n;
    Expr fn = cast<num_t>(Expr(n));

    // for (i = 0; i < n; i++)
    //     x[i] = 1 + (i / fn);
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

    // Vars and reduction domains
    Var i("i"), j("j");
    // Use the runtime extents of A to define the reduction ranges,
    // mirroring the original C loops over N and M.
    RDom r1(0, A.dim(0).extent(), "r1");  // corresponds to j in [0..n-1]
    RDom r2(0, A.dim(1).extent(), "r2");  // corresponds to i in [0..m-1]

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
    // -------------------------------------------------------------------------

    // Compute tmp once in full (like the first nested loop in the C code).
    tmp.compute_root();

    // For y's reduction, we want the outer loop to be over i (r2),
    // and the inner loop over j, to mirror:
    //   for (i = 0; i < M; i++)
    //     for (j = 0; j < N; j++)
    //       y[j] += A[i][j] * tmp[i];
    //
    // reorder(j, r2) => innermost j, outermost r2.
    y_func.update().reorder(j, r2);

    // Bind actual Buffers to the ImageParams
    A_param.set(A);
    x_param.set(x);

    // JIT-compile the kernel
    y_func.compile_jit();

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