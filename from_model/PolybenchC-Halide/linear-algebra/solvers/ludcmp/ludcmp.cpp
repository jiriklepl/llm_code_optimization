#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (PolyBench sizes & DATA_TYPE)
#include "defines.hpp"

// include benchmark-specific definitions (N, DATA_TYPE, etc)
#include "ludcmp.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize A, b, x, y the same way as the original C init_array().
 *
 * Original C:
 *   for i:
 *     x[i] = 0;
 *     y[i] = 0;
 *     b[i] = (i+1)/fn/2.0 + 4;
 *
 *   A initial lower-triangular construction, then:
 *     B = A * A^T;
 *     A = B;
 *
 * Layout note:
 *   We store A(row = i, col = j) at A(j, i), i.e. first dimension is column,
 *   second is row (column‑major). The LU and substitution code below treats
 *   A(j, i) consistently as A[i][j], so the physical layout is an
 *   implementation detail.
 */
static void init_array(int n,
                       Halide::Buffer<num_t, 2> &A,   // shape (n, n): (x=col, y=row)
                       Halide::Buffer<num_t, 1> &b,   // shape (n)
                       Halide::Buffer<num_t, 1> &x,   // shape (n)
                       Halide::Buffer<num_t, 1> &y) { // shape (n)
    Var i("i"), col("col"), row("row");

    // Initialize x and y to 0, b[i] = (i+1)/n/2 + 4
    Func init_x("init_x"), init_y("init_y"), init_b("init_b");

    Expr n_expr = Expr(n);

    init_x(i) = cast<num_t>(Expr(0.0));
    init_y(i) = cast<num_t>(Expr(0.0));

    // term = (i+1)/n/2 + 4
    Expr term1 = cast<num_t>(i + 1) / cast<num_t>(n_expr);
    Expr term2 = term1 / cast<num_t>(Expr(2.0));
    Expr term3 = term2 + cast<num_t>(Expr(4.0));
    init_b(i) = term3;

    init_x.realize(x);
    init_y.realize(y);
    init_b.realize(b);

    // Initial lower-triangular A, with
    //   A[i][j] = (-j % n) / n + 1  for j <= i,
    //            0                  otherwise,
    //   and A[i][i] = 1.
    //
    // We store A(row=i, col=j) at A(j, i).
    Func A_init("A_init");

    Expr neg_mod = -(col % n);
    Expr base = cast<num_t>(neg_mod) / cast<num_t>(n_expr) + cast<num_t>(Expr(1.0));

    A_init(col, row) =
        select(col == row,
               cast<num_t>(Expr(1.0)),                      // diagonal = 1
               select(col <= row,
                      base,                                 // lower triangle
                      cast<num_t>(Expr(0.0))));             // upper triangle = 0

    // Make the matrix positive semi-definite: B = A * A^T, then A = B.
    //
    // B[r][s] = sum_t A[r][t] * A[s][t]
    // In our layout: A(r, t) is A[row=r, col=t], so:
    // B(row=r, col=s) = sum_t A_init(t, r) * A_init(t, s)
    Func SPD("SPD");
    RDom t_dom(0, n, "t_dom");

    SPD(col, row) = cast<num_t>(Expr(0.0));
    SPD(col, row) += A_init(t_dom, row) * A_init(t_dom, col);

    // Realize SPD into A buffer (shape n x n).
    SPD.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Buffers:
    // A: 2D (n x n), first dimension = column (j), second = row (i)
    // b, x, y: 1D (n)
    Halide::Buffer<num_t, 2> A(n, n);
    Halide::Buffer<num_t, 1> b(n), x(n), y(n);

    // Initialize data (outside of timed region)
    init_array(n, A, b, x, y);

    // Extra 1D buffer to hold reciprocals of diagonal entries of U.
    // This replaces many divisions in the L‑update and back‑substitution
    // with multiplications, which is significantly faster on modern CPUs.
    Halide::Buffer<num_t, 1> diag_inv(n);

    // Set up Halide parameters and ImageParams for the kernel.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam b_param(type_of<num_t>(), 1, "b_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");
    ImageParam y_param(type_of<num_t>(), 1, "y_param");
    ImageParam diag_param(type_of<num_t>(), 1, "diag_param");

    Param<int> pi("pi");  // current row index i
    Param<int> pj("pj");  // current column index j
    Param<int> pn("pn");  // n (for convenience in ranges)

    A_param.set(A);
    b_param.set(b);
    x_param.set(x);
    y_param.set(y);
    diag_param.set(diag_inv);
    pn.set(n);

    // Target and vector width for scheduling
    Target target = get_jit_target_from_environment();
    Type num_type = type_of<num_t>();
    int vec_width = target.natural_vector_size(num_type);
    if (vec_width <= 0) {
        vec_width = 1;
    }

    // --------------------------------------------------------------------
    // Halide helpers for LU decomposition and forward/back substitution
    // --------------------------------------------------------------------

    // 1) L-part update:
    // For 0 <= j < i:
    //   w = A[i][j] - sum_{k=0..j-1} A[i][k] * A[k][j];
    //   A[i][j] = w / A[j][j];
    //
    // We rewrite A[i][j] = w * diag_inv[j], with diag_inv[j] = 1/A[j][j],
    // which preserves the mathematical result up to normal FP rounding.
    //
    // In our layout: A[i][j] == A_param(j, i).
    Func update_L("update_L");
    RDom rL(0, pj, "rL");  // k in [0, j-1]
    {
        Expr sum_L = sum(A_param(rL, pi) * A_param(pj, rL));
        Expr w = A_param(pj, pi) - sum_L;
        update_L() = w * diag_param(pj);
    }

    // Simple schedule: vectorize the reduction over k when it is large enough.
    // This turns the inner dot-product into SIMD code.
    update_L.update().vectorize(rL, vec_width);

    // 2) U-part update for an entire row i (all j >= i) in one Halide call.
    //
    // For j >= i:
    //   w = A[i][j] - sum_{k=0..i-1} A[i][k] * A[k][j];
    //   A[i][j] = w;
    //
    // In our layout: A[i][j] == A_param(j, i).
    //
    // We compute the tail of the row, j = i..n-1, by using a local
    // coordinate jt = 0..(n-i-1) and mapping j_global = jt + i:
    //   w(jt) = A_param(jt + i, i) -
    //           sum_{k=0..i-1} A_param(k, i) * A_param(jt + i, k)
    //   update_U_row(jt) = w(jt)
    //
    // This lets Halide compute a whole row segment at once, which we can
    // vectorize across jt.
    Func update_U_row("update_U_row");
    Var jt("jt");
    RDom rU(0, pi, "rU");  // k in [0, i-1]
    {
        Expr j_global = jt + pi;
        Expr sum_U = sum(A_param(rU, pi) * A_param(j_global, rU));
        update_U_row(jt) = A_param(j_global, pi) - sum_U;
    }

    // Schedule for U-row update:
    // - Compute at root, as a stand-alone kernel.
    // - Vectorize across jt (columns in the tail of the row) so that
    //   multiple U[i][j] values are updated in parallel.
    update_U_row.compute_root()
                .vectorize(jt, vec_width);

    // 3) Forward substitution for y:
    // For i = 0..n-1:
    //   y[i] = b[i] - sum_{j=0..i-1} A[i][j] * y[j];
    //
    // In our layout: A[i][j] == A_param(j, i).
    Func compute_y("compute_y");
    RDom rY(0, pi, "rY");  // j in [0, i-1]
    {
        Expr sum_Y = sum(A_param(rY, pi) * y_param(rY));
        compute_y() = b_param(pi) - sum_Y;
    }
    // (This reduction is over a single scalar result, so we leave it
    //  with the default schedule; the sum helper will generate an
    //  efficient inner loop.)

    // 4) Back substitution for x:
    // For i = n-1..0:
    //   x[i] = (y[i] - sum_{j=i+1..n-1} A[i][j] * x[j]) / A[i][i];
    //
    // We rewrite using diag_inv[i] = 1 / A[i][i]:
    //   x[i] = (y[i] - sum_{j=i+1..n-1} A[i][j] * x[j]) * diag_inv[i]
    //
    // In our layout: A[i][j] == A_param(j, i).
    Func compute_x("compute_x");
    {
        // Domain j = i+1 .. n-1  -> RDom(pi+1, n-(pi+1))
        RDom rX(pi + 1, pn - (pi + 1), "rX");
        Expr sum_X = sum(A_param(rX, pi) * x_param(rX));
        compute_x() = (y_param(pi) - sum_X) * diag_param(pi);
    }

    // --------------------------------------------------------------------
    // Compile the helper pipelines once.
    // --------------------------------------------------------------------
    update_L.compile_jit(target);
    update_U_row.compile_jit(target);
    compute_y.compile_jit(target);
    compute_x.compile_jit(target);

    // --------------------------------------------------------------------
    // Run the kernel and measure its time.
    // --------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // Small scalar buffers to avoid re-allocating for every realize().
    // 0-D buffers are used here: a single scalar each.
    Halide::Buffer<num_t> scalar_L;
    Halide::Buffer<num_t> scalar_y;
    Halide::Buffer<num_t> scalar_x;

    // LU decomposition: in-place factorization of A into L and U.
    //
    // A is stored as column-major A(j, i) representing A[i][j].
    // We follow the C algorithm exactly, but:
    //   - use diag_inv[j] to replace divisions w / A[j][j] with w * diag_inv[j]
    //   - compute the whole U row segment j = i..n-1 at once with update_U_row
    //     to exploit vectorization and better data locality.
    for (int i = 0; i < n; i++) {
        // L-part: 0 <= j < i
        for (int j = 0; j < i; j++) {
            pi.set(i);
            pj.set(j);
            update_L.realize(scalar_L);
            // A[i][j] is stored at A(j, i).
            A(j, i) = scalar_L();
        }

        // U-part: j >= i, computed in one Halide call for the tail of the row.
        {
            pi.set(i);
            // Number of columns in the tail [i, n)
            int tail_len = n - i;
            // Realize U[i][j] for j in [i, n) as jt in [0, tail_len)
            Halide::Buffer<num_t> tail = update_U_row.realize({tail_len});
            // Copy back to A. tail(jt) corresponds to column j = i + jt.
            for (int jt = 0; jt < tail_len; ++jt) {
                int j = i + jt;
                A(j, i) = tail(jt);
            }
        }

        // Maintain reciprocal of the diagonal A[i][i] for later L-updates
        // (rows below this one) and back substitution. This reduces the
        // number of divisions from O(n^2) to O(n).
        diag_inv(i) = (num_t)1.0 / A(i, i);
    }

    // Forward substitution to compute y: L * y = b
    for (int i = 0; i < n; i++) {
        pi.set(i);
        compute_y.realize(scalar_y);
        y(i) = scalar_y();
    }

    // Back substitution to compute x: U * x = y
    for (int i = n - 1; i >= 0; i--) {
        pi.set(i);
        compute_x.realize(scalar_x);
        x(i) = scalar_x();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result vector x, similar to print_array().
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            if (i % 20 == 0) {
                std::cerr << '\n';
            }
            std::cerr << x(i) << ' ';
        }
        std::cerr << '\n';
    }

    // Print timing (seconds)
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}