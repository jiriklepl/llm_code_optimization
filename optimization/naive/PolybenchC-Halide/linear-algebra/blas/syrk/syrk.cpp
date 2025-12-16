#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "syrk.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Array initialization, translated from the original C version.
 * We use Halide to initialize A and C, and schedule the initialization
 * for good cache use and SIMD.
 */
static
void init_array(int n, int m,
                num_t *alpha,
                num_t *beta,
                Halide::Buffer<num_t, 2> C,
                Halide::Buffer<num_t, 2> A) {

    *alpha = (num_t)1.5;
    *beta  = (num_t)1.2;

    Var i("i"), j("j");
    Func init_A("init_A"), init_C("init_C");

    // We'll treat A as a Halide image of shape (m, n), indexed as A(j, i):
    //   C version: A[i][j]  with i in [0,n), j in [0,m)
    //   Halide:    A(j, i)  with j -> column (0..m-1), i -> row (0..n-1)
    //
    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < m; j++)
    //     A[i][j] = ((i*j+1)%n) / n;
    //
    // Map into Halide:
    //   i -> Var i (rows), j -> Var j (cols)
    Expr n_expr = Expr(n);
    Expr m_expr = Expr(m);

    Expr a_val = ((i * j + 1) % n_expr);
    init_A(j, i) = cast<num_t>(a_val) / cast<num_t>(n_expr);

    // C is N x N in C as C[i][j], mapped to Buffer C(j, i) shape (n, n).
    //
    // Original C:
    // for (i = 0; i < n; i++)
    //   for (j = 0; j < n; j++)
    //     C[i][j] = ((i*j+2)%m) / m;
    //
    Expr c_val = ((i * j + 2) % m_expr);
    init_C(j, i) = cast<num_t>(c_val) / cast<num_t>(m_expr);

    // Simple CPU schedule: row-parallel, SIMD across columns.
    Target target = get_jit_target_from_environment();
    const int vec = target.natural_vector_size(type_of<num_t>());

    init_A
        .compute_root()
        .reorder(j, i)       // j = x (fast), i = y (slow)
        .vectorize(j, vec)
        .parallel(i);

    init_C
        .compute_root()
        .reorder(j, i)
        .vectorize(j, vec)
        .parallel(i);

    // Realize into the provided buffers.
    init_A.realize(A);
    init_C.realize(C);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;
    int m = M;

    // Scalars
    num_t alpha;
    num_t beta;

    // Buffers:
    // C is N x N in C as C[i][j], map to Buffer C(j, i) => shape (n, n)
    // A is N x M in C as A[i][j], map to Buffer A(j, i) => shape (m, n)
    Halide::Buffer<num_t, 2> C(n, n);
    Halide::Buffer<num_t, 2> A(m, n);

    // ImageParams corresponding to C and A
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam A_param(type_of<num_t>(), 2, "A_param");

    // Initialize data using Halide
    init_array(n, m, &alpha, &beta, C, A);

    // Bind ImageParams to the initialized buffers
    C_param.set(C);
    A_param.set(A);

    // Scalar parameters for alpha and beta.
    // Using Param allows us to compile once and vary values cheaply.
    Param<num_t> alpha_p("alpha_p"), beta_p("beta_p");
    alpha_p.set(alpha);
    beta_p.set(beta);

    // Halide Vars and RDom
    Var i("i"), j("j");

    // We will use:
    //  - one RDom over j for scaling by beta (only j <= i),
    //  - one RDom over k for the SYRK accumulation.
    RDom rj(0, n, "rj");    // for scaling C's lower triangle
    RDom rk(0, m, "rk");    // k over M (first dimension of A)

    // BLAS SYRK:
    //   C := alpha * A * A^T + beta * C
    //
    // Original C kernel:
    // for (i = 0; i < _PB_N; i++) {
    //   for (j = 0; j <= i; j++)
    //     C[i][j] *= beta;
    //   for (k = 0; k < _PB_M; k++) {
    //     for (j = 0; j <= i; j++)
    //       C[i][j] += alpha * A[i][k] * A[j][k];
    //   }
    // }
    //
    // Mapping to Halide:
    //   C[i][j]     -> C_param(j, i)
    //   A[i][k]     -> A_param(k, i)
    //   A[j][k]     -> A_param(k, j)
    //
    // We will:
    //   1) Copy C into a Func syrk(j, i)
    //   2) Scale C's lower triangle by beta (using a predicated RDom over j)
    //   3) Add alpha * A(i,k) * A(j,k) into the lower triangle
    //
    // The upper triangle (j > i) is left unchanged, matching the original code.

    Func syrk("syrk");

    // 1) Pure definition: copy original C.
    syrk(j, i) = C_param(j, i);

    // 2) Scale lower triangle by beta: for each row i, scale entries j in [0, i].
    //    We do this via an RDom over j with a predicate (triangle domain),
    //    which avoids unnecessary multiplies for j > i and keeps upper
    //    triangle untouched.
    rj.where(rj <= i);  // Only operate on j <= i
    syrk(rj, i) = beta_p * syrk(rj, i);

    // 3) SYRK accumulation over k:
    //    For all (i, j) we conceptually compute:
    //      if (j <= i) C[i][j] += alpha * sum_k A[i][k] * A[j][k]
    //      else        C[i][j] unchanged
    //
    //    We express this as a reduction over rk with a select on j <= i.
    syrk(j, i) += select(j <= i,
                         alpha_p * A_param(rk, i) * A_param(rk, j),
                         cast<num_t>(0));

    // ------------------------------------------------------------------
    // Scheduling
    // ------------------------------------------------------------------
    //
    // Layout reminders:
    //   - syrk(j, i): j is x (fast/contiguous), i is y (slow).
    //   - A_param(k, i): k is x (fast), i is y.
    //
    // Goals:
    //   - Iterate j as the innermost loop for good spatial locality in C.
    //   - Parallelize across rows (i) to exploit multiple cores.
    //   - Vectorize across j where possible.
    //   - For the beta scaling step, only touch the lower triangle.
    //   - For the main reduction, keep k as the reduction dimension and
    //     avoid excessive complexity (no rfactor here for clarity).

    Target target = get_jit_target_from_environment();
    const int vec = target.natural_vector_size(type_of<num_t>());

    // Make syrk a top-level (compute_root) Func.
    syrk
        .compute_root()
        .reorder(j, i)       // j (x) innermost, i (y) outermost
        .vectorize(j, vec)   // SIMD across columns
        .parallel(i);        // parallelize across rows

    // Schedule for update 0: scaling by beta over lower triangle.
    {
        RVar rjv = rj.x;
        syrk.update(0)
            .reorder(rjv, i)     // rj (j index) innermost
            .vectorize(rjv, vec)
            .parallel(i);
    }

    // Schedule for update 1: SYRK accumulation over k.
    // We keep j innermost and vectorized; k is the inner reduction loop.
    {
        syrk.update(1)
            .reorder(j, rk, i)   // j innermost, then k, then i
            .vectorize(j, vec)
            .parallel(i);
    }

    // Compile the kernel to machine code ahead of timing.
    syrk.compile_jit(target);

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: write result into C (in-place semantics w.r.t the original code)
    syrk.realize(C);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<long double>(end - start);

    // Print results (similar spirit to the original print_array), guarded
    // the same way as in the GEMM example.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                // C array C[ii][jj] -> Buffer C(jj, ii)
                std::cerr << C(jj, ii) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}