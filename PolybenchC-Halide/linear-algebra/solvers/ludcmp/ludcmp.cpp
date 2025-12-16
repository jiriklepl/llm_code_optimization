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

    // Initial lower-triangular A, with A[i][j] = (-j % n) / n + 1 for j <= i, 0 otherwise, and A[i][i] = 1.
    //
    // We store A(row=i, col=j) at A(j, i), i.e., first dimension is column, second is row.
    Func A_init("A_init");

    // (-j % n) / n + 1  -> implement as - (j % n) / n + 1
    Expr neg_mod = -(col % n);
    Expr base = cast<num_t>(neg_mod) / cast<num_t>(n_expr) + cast<num_t>(Expr(1.0));

    // col = j, row = i
    A_init(col, row) =
        select(col == row,
               cast<num_t>(Expr(1.0)),                      // diagonal = 1
               select(col <= row,
                      base,                                 // lower triangle (including diagonal before override)
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

    // Initialize data
    init_array(n, A, b, x, y);

    // Set up Halide parameters and ImageParams for the kernel.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam b_param(type_of<num_t>(), 1, "b_param");
    ImageParam x_param(type_of<num_t>(), 1, "x_param");
    ImageParam y_param(type_of<num_t>(), 1, "y_param");

    Param<int> pi("pi");  // current i
    Param<int> pj("pj");  // current j
    Param<int> pn("pn");  // n (for convenience in ranges)

    A_param.set(A);
    b_param.set(b);
    x_param.set(x);
    y_param.set(y);
    pn.set(n);

    // --------------------------------------------------------------------
    // Halide helpers for the LU decomposition and forward/back substitution
    // --------------------------------------------------------------------

    // 1) L-part update:
    // For j < i:
    //   w = A[i][j];
    //   for k = 0..j-1: w -= A[i][k] * A[k][j];
    //   A[i][j] = w / A[j][j];
    //
    // In our layout: A[i][j] == A_param(j, i).
    // We compute the new scalar A[i][j] as:
    //   A_param(j, i) - sum_{k=0..j-1} A_param(k, i) * A_param(j, k)
    // then divide by A_param(j, j).
    Func update_L("update_L");
    {
        RDom rL(0, pj, "rL");
        Expr sum_L = sum(A_param(rL, pi) * A_param(pj, rL));
        update_L() = (A_param(pj, pi) - sum_L) / A_param(pj, pj);
    }

    // 2) U-part update:
    // For j >= i:
    //   w = A[i][j];
    //   for k = 0..i-1: w -= A[i][k] * A[k][j];
    //   A[i][j] = w;
    //
    // In our layout: A[i][j] == A_param(j, i).
    // new A[i][j] = A_param(j, i) - sum_{k=0..i-1} A_param(k, i) * A_param(j, k).
    Func update_U("update_U");
    {
        RDom rU(0, pi, "rU");
        Expr sum_U = sum(A_param(rU, pi) * A_param(pj, rU));
        update_U() = A_param(pj, pi) - sum_U;
    }

    // 3) Forward substitution for y:
    // For i = 0..n-1:
    //   w = b[i];
    //   for j = 0..i-1: w -= A[i][j] * y[j];
    //   y[i] = w;
    //
    // y[i] = b[i] - sum_{j=0..i-1} A[i][j] * y[j].
    // In our layout: A[i][j] == A_param(j, i).
    Func compute_y("compute_y");
    {
        RDom rY(0, pi, "rY");
        Expr sum_Y = sum(A_param(rY, pi) * y_param(rY));
        compute_y() = b_param(pi) - sum_Y;
    }

    // 4) Back substitution for x:
    // For i = n-1..0:
    //   w = y[i];
    //   for j = i+1..n-1: w -= A[i][j] * x[j];
    //   x[i] = w / A[i][i];
    //
    // x[i] = (y[i] - sum_{j=i+1..n-1} A[i][j] * x[j]) / A[i][i].
    // In our layout: A[i][j] == A_param(j, i).
    Func compute_x("compute_x");
    {
        // Domain j = i+1 .. n-1  -> RDom(pi+1, n-(pi+1))
        RDom rX(pi + 1, pn - (pi + 1), "rX");
        Expr sum_X = sum(A_param(rX, pi) * x_param(rX));
        compute_x() = (y_param(pi) - sum_X) / A_param(pi, pi);
    }

    // Compile the small helper pipelines once.
    update_L.compile_jit();
    update_U.compile_jit();
    compute_y.compile_jit();
    compute_x.compile_jit();

    // ------------------------------------
    // Run the kernel and measure its time.
    // ------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // LU decomposition
    for (int i = 0; i < n; i++) {
        // L-part: j < i
        for (int j = 0; j < i; j++) {
            pi.set(i);
            pj.set(j);
            Halide::Buffer<num_t> tmp = update_L.realize();
            // A[i][j] -> A(j, i)
            A(j, i) = tmp();
        }

        // U-part: j >= i
        for (int j = i; j < n; j++) {
            pi.set(i);
            pj.set(j);
            Halide::Buffer<num_t> tmp = update_U.realize();
            A(j, i) = tmp();
        }
    }

    // Forward substitution to compute y
    for (int i = 0; i < n; i++) {
        pi.set(i);
        Halide::Buffer<num_t> tmp = compute_y.realize();
        y(i) = tmp();
    }

    // Back substitution to compute x
    for (int i = n - 1; i >= 0; i--) {
        pi.set(i);
        Halide::Buffer<num_t> tmp = compute_x.realize();
        x(i) = tmp();
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