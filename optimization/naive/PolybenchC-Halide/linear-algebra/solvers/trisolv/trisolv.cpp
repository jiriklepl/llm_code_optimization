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

/* Array initialization translated to Halide, with a simple CPU schedule. */
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
    // Remember: L is stored as L(j, i) to make j the fast-varying dimension.
    Expr n_expr = cast<num_t>(Expr(n));
    Expr two    = cast<num_t>(Expr(2));

    init_L(j, i) = select(j <= i,
                          (cast<num_t>(i + n - j + 1) * two) / n_expr,
                          cast<num_t>(Expr(0)));

    // ------------------------------------------------------------------
    // Simple but effective schedule for initialization:
    //  - Vectorize along the innermost dimension.
    //  - Parallelize along the outer dimension where cheap and safe.
    // ------------------------------------------------------------------
    Target target = get_host_target();
    const int vec = target.natural_vector_size(type_of<num_t>());

    // 1D buffers: parallelize and vectorize over i
    init_x
        .compute_root()
        .vectorize(i, vec)
        .parallel(i);

    init_b
        .compute_root()
        .vectorize(i, vec)
        .parallel(i);

    // 2D L: parallelize over rows (i), vectorize over columns (j).
    init_L
        .compute_root()
        .parallel(i)
        .vectorize(j, vec);

    // Materialize into the output Buffers.
    init_x.realize(x);
    init_b.realize(b);
    init_L.realize(L);
}

int main(int argc, char *argv[]) {
    // problem size
    int n = N;

    // Buffers:
    // L: original is L[i][j]; we store as L(j, i) so that the first
    // dimension (j) is the fast-varying one (better cache locality).
    Halide::Buffer<num_t, 2> L(n, n);
    Halide::Buffer<num_t, 1> x(n);
    Halide::Buffer<num_t, 1> b(n);

    // ImageParams corresponding to L, x, b
    ImageParam L_param(type_of<num_t>(), 2, "L_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");
    ImageParam b_param(type_of<num_t>(), 1, "b_param");

    // Scalar Param for the current row index i
    Param<int> i_param("i_param");

    // initialize data (not included in the timed region)
    init_array(n, L, x, b);

    // Bind ImageParams to our Buffers once; we will just change i_param in the loop.
    L_param.set(L);
    x_param.set(x);
    b_param.set(b);

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
    // -----------------------------------------------------------------

    Func xi("xi");         // accumulates b[i] - sum_{j<i} L[i][j]*x[j]
    Func xi_div("xi_div"); // final x[i] = xi / L[i][i]

    // RDom over j = 0 .. i_param-1 (runtime extent depending on i_param).
    RDom r(0, i_param, "r");

    // Start from b[i]
    xi() = b_param(i_param);
    // Subtract contributions L[i][j] * x[j]
    // Recall: L[i][j] is stored as L_param(j, i_param)
    xi() -= L_param(r, i_param) * x_param(r);

    // Divide by the diagonal entry L[i][i] (L_param(i, i))
    xi_div() = xi() / L_param(i_param, i_param);

    // -----------------------------------------------------------------
    // JIT-compile once with an explicit target, so that all subsequent
    // realizations reuse the compiled code (no re-JIT inside the loop).
    // -----------------------------------------------------------------
    Target target = get_host_target();
    xi_div.compile_jit(target);

    // Reuse a pre-allocated scalar Buffer for xi_div() to avoid
    // per-iteration allocation overhead.
    Halide::Buffer<num_t> xi_scalar = Halide::Buffer<num_t>::make_scalar();

    // -----------------------------------------------------------------
    // Run the triangular solve, timing only the kernel-like part:
    // -----------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        // Set current row index
        i_param.set(i);

        // Realize scalar xi_div() for this i into the pre-allocated buffer
        xi_div.realize(xi_scalar);

        // Update x[i] on the host
        x(i) = xi_scalar();
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