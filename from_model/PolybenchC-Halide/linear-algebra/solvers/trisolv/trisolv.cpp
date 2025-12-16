#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions (e.g. N, DATA_TYPE)
#include "defines.hpp"

// include benchmark-specific definitions (same role as trisolv.h)
#include "trisolv.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/* Array initialization translated to Halide, with a simple SIMD-friendly schedule. */
static
void init_array(int n,
                Halide::Buffer<num_t, 2> L,  // logical L[i][j] stored as L(j, i)
                Halide::Buffer<num_t, 1> x,
                Halide::Buffer<num_t, 1> b) {
    Var i("i"), j("j");
    Func init_x("init_x"), init_b("init_b"), init_L("init_L");

    // x[i] = -999;
    init_x(i) = cast<num_t>(Expr(-999));

    // b[i] = i;
    init_b(i) = cast<num_t>(i);

    // L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n for j <= i; 0 elsewhere (unused in kernel).
    // Remember: L is stored as L(j, i).
    Expr n_expr = cast<num_t>(Expr(n));
    Expr two    = cast<num_t>(Expr(2));

    init_L(j, i) = select(j <= i,
                          (cast<num_t>(i + n - j + 1) * two) / n_expr,
                          cast<num_t>(Expr(0)));

    // -----------------------------------------------------------------
    // Simple initialization schedule:
    //  - X and b: just vectorize over i.
    //  - L: parallelize over rows (i) and vectorize across j, which is
    //    the fast-varying dimension in memory.
    // This keeps init cheap; it runs once so we don't over‑engineer it.
    // -----------------------------------------------------------------
    Target t = get_host_target();
    const int vec = std::max(1, t.natural_vector_size(type_of<num_t>()));

    init_x.compute_root()
          .vectorize(i, vec);

    init_b.compute_root()
          .vectorize(i, vec);

    init_L.compute_root()
          .parallel(i)
          .vectorize(j, vec);

    init_x.realize(x);
    init_b.realize(b);
    init_L.realize(L);
}

int main(int argc, char *argv[]) {
    // problem size
    int n = N;

    // Buffers:
    // L: original is L[i][j]; we store as L(j, i) so that the first
    // dimension (j) is the fast-varying one.
    Halide::Buffer<num_t, 2> L(n, n);
    Halide::Buffer<num_t, 1> x(n);
    Halide::Buffer<num_t, 1> b(n);

    // ImageParams corresponding to L, x, b
    ImageParam L_param(type_of<num_t>(), 2, "L_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");
    ImageParam b_param(type_of<num_t>(), 1, "b_param");

    // Scalar Param for the current row index i
    Param<int> i_param("i_param");

    // initialize data
    init_array(n, L, x, b);

    // -----------------------------------------------------------------
    // Halide formulation of the inner computation:
    //
    // For a fixed i:
    //   x[i] = b[i];
    //   for (j = 0; j < i; j++)
    //       x[i] -= L[i][j] * x[j];
    //   x[i] = x[i] / L[i][i];
    //
    // We express the j-loop as a reduction over RDom, and drive the
    // i-loop from the host, updating x one element at a time.
    //
    // The outer i-loop is inherently sequential due to the triangular
    // dependency x[i] depending on x[0..i-1]. The inner reduction over
    // j is purely a dot product, which we can tile and vectorize.
    // -----------------------------------------------------------------

    Func xi("xi");        // accumulates b[i] - sum_{j<i} L[i][j]*x[j]
    Func xi_div("xi_div");// final x[i] = xi / L[i][i]

    // RDom over j = 0 .. i_param-1
    RDom r(0, i_param, "r");

    // Start from b[i]
    xi() = b_param(i_param);

    // Subtract contributions L[i][j] * x[j]
    // Recall: L[i][j] is stored as L_param(j, i_param)
    xi() -= L_param(r, i_param) * x_param(r);

    // Divide by the diagonal entry L[i][i] (L_param(i, i))
    xi_div() = xi() / L_param(i_param, i_param);

    // -----------------------------------------------------------------
    // Scheduling for the inner reduction:
    //
    //  - We keep the host-driven i-loop as is (sequential).
    //  - Inside the Halide kernel, we tile and vectorize the reduction
    //    over r (j) to improve locality and use SIMD.
    //
    //  L is laid out with j as the fast-varying dimension (L(j, i)),
    //  and x(j) is contiguous, so vectorizing over r gives unit-stride
    //  loads for both L and x.
    // -----------------------------------------------------------------
    Target t = get_host_target();
    const int vec = std::max(1, t.natural_vector_size(type_of<num_t>()));
    const int r_tile = 128; // tile size along j; fits easily in L1/L2 for typical N

    RVar r_outer("r_outer"), r_inner("r_inner");

    // Compute the scalar reduction result at root.
    xi.compute_root();

    // Split r into tiles and vectorize the inner part.
    //
    // Conceptually for each call (fixed i_param):
    //   for r_outer = 0 .. ceil(i_param / r_tile)-1:
    //     for r_inner = 0 .. r_tile-1:
    //       j = r_outer * r_tile + r_inner;
    //       if (j < i_param)  xi -= L(j,i)*x(j);
    //
    // Halide's default tail strategy guards out-of-range iterations.
    xi.update()
      .split(r, r_outer, r_inner, r_tile)
      .vectorize(r_inner, vec)
      .unroll(r_inner, 2);

    // The final divide is a single scalar op; compute at root as well.
    xi_div.compute_root();

    // Bind ImageParams to our Buffers (x will be updated in-place by host).
    L_param.set(L);
    x_param.set(x);
    b_param.set(b);

    // JIT-compile the pipeline once
    xi_div.compile_jit();

    // Preallocate a 1-element buffer for the scalar result to avoid
    // per-iteration heap allocation inside realize().
    Halide::Buffer<num_t> xi_buf(1);

    // -----------------------------------------------------------------
    // Run the triangular solve, timing only the kernel-like part:
    // -----------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        // Set current row index
        i_param.set(i);

        // Realize scalar xi_div() for this i into the reusable buffer.
        xi_div.realize(xi_buf);

        // Update x[i] on the host
        x(i) = xi_buf(0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -----------------------------------------------------------------
    // Print results (similar style to example/gemm.cpp)
    // -----------------------------------------------------------------
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            std::cerr << x(i) << '\n';
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}