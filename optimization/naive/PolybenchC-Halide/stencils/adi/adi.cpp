#include <chrono>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "Halide.h"

// Common PolyBench-style definitions (DATA_TYPE, etc.)
#include "defines.hpp"

// Benchmark-specific definitions (N, TSTEPS, etc.)
#include "adi.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/*
 * Initialize array u:
 *
 * Original C:
 *   for (i = 0; i < n; i++)
 *     for (j = 0; j < n; j++)
 *       u[i][j] = (DATA_TYPE)(i + n - j) / n;
 *
 * C array u[i][j] is mapped to Halide Buffer u(j, i).
 */
static void init_array(int n, Halide::Buffer<num_t, 2> u) {
    Var x("x"), y("y"); // x == j (fast), y == i (slow)
    Func init_u("init_u");

    // u[i][j] = (i + n - j) / n  ==>  u(j, i) = ...
    init_u(x, y) = cast<num_t>(y + n - x) / cast<num_t>(n);

    // Simple but effective schedule: vectorize across x (fast dim) and
    // parallelize across y (slow dim).
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    init_u
        .vectorize(x, vec_width)
        .parallel(y);

    init_u.realize(u);
}

/*
 * Main computational kernel translated to C++ operating on Halide::Buffer.
 *
 * Original signature:
 *
 *   void kernel_adi(int tsteps, int n,
 *                   DATA_TYPE u[N][N],
 *                   DATA_TYPE v[N][N],
 *                   DATA_TYPE p[N][N],
 *                   DATA_TYPE q[N][N]);
 *
 * Mapping of C indices A[i][j] -> Buffer A(j, i).
 *
 * NOTE: The original algorithm performs, for each timestep:
 *   1) A "column sweep" (solving a tridiagonal system along one dimension)
 *   2) A "row sweep"    (solving a tridiagonal system along the other)
 *
 * Along the sweep dimension there are loop-carried dependencies
 * (Thomas algorithm), which cannot be expressed directly as a single
 * Halide reduction (it would require f(j) depending on f(j-1)).
 *
 * To still leverage Halide, we:
 *   - Keep the sweep dimension (j) as a host loop (sequential).
 *   - Use Halide Funcs to compute, in parallel over the orthogonal
 *     dimension (i), each *sweep step* for a fixed j.
 *
 * This improves locality and enables vectorization/parallelism over i,
 * while preserving the exact numerical semantics of the original code.
 */
static void kernel_adi(int tsteps,
                       int n,
                       Halide::Buffer<num_t, 2> u,
                       Halide::Buffer<num_t, 2> v,
                       Halide::Buffer<num_t, 2> p,
                       Halide::Buffer<num_t, 2> q) {
    // Scalar parameters (identical to original C version).
    num_t DX   = num_t(1.0) / num_t(n);
    num_t DY   = num_t(1.0) / num_t(n);
    num_t DT   = num_t(1.0) / num_t(tsteps);
    num_t B1   = num_t(2.0);
    num_t B2   = num_t(1.0);
    num_t mul1 = B1 * DT / (DX * DX);
    num_t mul2 = B2 * DT / (DY * DY);

    num_t a_val = -mul1 / num_t(2.0);
    num_t b_val = num_t(1.0) + mul1;
    num_t c_val = a_val;
    num_t d_val = -mul2 / num_t(2.0);
    num_t e_val = num_t(1.0) + mul2;
    num_t f_val = d_val;

    // Precompute some frequently used combinations to avoid mixing
    // C++ literals with Halide Exprs of unknown type.
    num_t one_plus_2d_val = num_t(1.0) + num_t(2.0) * d_val;
    num_t one_plus_2a_val = num_t(1.0) + num_t(2.0) * a_val;

    // -----------------------------
    // Halide setup for sweep steps
    // -----------------------------

    // Vars and parameters shared across the step pipelines
    Var i("i");
    Param<int> j_param("j_param");

    // ImageParams wrap the current full 2D arrays. They are *views*
    // onto the existing Buffers; modifying the Buffers updates what
    // the Funcs see.
    ImageParam U_img(type_of<num_t>(), 2, "U_img");
    ImageParam V_img(type_of<num_t>(), 2, "V_img");
    ImageParam P_img(type_of<num_t>(), 2, "P_img");
    ImageParam Q_img(type_of<num_t>(), 2, "Q_img");

    // Scalar Params for coefficients
    Param<num_t> a_p("a_p"), b_p("b_p"), c_p("c_p");
    Param<num_t> d_p("d_p"), e_p("e_p"), f_p("f_p");
    Param<num_t> one_plus_2d_p("one_plus_2d_p");
    Param<num_t> one_plus_2a_p("one_plus_2a_p");

    // ---------------- Column sweep step: forward (compute p[j,*], q[j,*]) ---------
    //
    // For fixed j, and for all interior i:
    //
    //   denom = a * p(j-1, i) + b;
    //   p(j, i) = -c / denom;
    //
    //   num = -d * u(i-1, j)
    //         + (1 + 2*d) * u(i, j)
    //         - f * u(i+1, j)
    //         - a * q(j-1, i);
    //   q(j, i) = num / denom;
    //
    // We compute (p(j, i), q(j, i)) as a Tuple over i.
    Func col_forward("col_forward");
    {
        Expr denom = a_p * P_img(j_param - 1, i) + b_p;

        Expr num = -d_p * U_img(i - 1, j_param) +
                   one_plus_2d_p * U_img(i, j_param) -
                   f_p * U_img(i + 1, j_param) -
                   a_p * Q_img(j_param - 1, i);

        Expr p_new = -c_p / denom;
        Expr q_new = num / denom;

        col_forward(i) = Tuple(p_new, q_new);
    }

    // ---------------- Column sweep step: backward (compute v[*,j]) ---------------
    //
    // For fixed j, and for all interior i:
    //
    //   v(i, j) = p(j, i) * v(i, j+1) + q(j, i);
    //
    Func col_backward("col_backward");
    {
        Expr v_new = P_img(j_param, i) * V_img(i, j_param + 1) +
                     Q_img(j_param, i);
        col_backward(i) = v_new;
    }

    // ---------------- Row sweep step: forward (compute p[j,*], q[j,*]) -----------
    //
    // For fixed j, and for all interior i:
    //
    //   denom = d * p(j-1, i) + e;
    //   p(j, i) = -f / denom;
    //
    //   num = -a * v(j, i-1)
    //         + (1 + 2*a) * v(j, i)
    //         - c * v(j, i+1)
    //         - d * q(j-1, i);
    //   q(j, i) = num / denom;
    //
    Func row_forward("row_forward");
    {
        Expr denom = d_p * P_img(j_param - 1, i) + e_p;

        Expr num = -a_p * V_img(j_param, i - 1) +
                   one_plus_2a_p * V_img(j_param, i) -
                   c_p * V_img(j_param, i + 1) -
                   d_p * Q_img(j_param - 1, i);

        Expr p_new = -f_p / denom;
        Expr q_new = num / denom;

        row_forward(i) = Tuple(p_new, q_new);
    }

    // ---------------- Row sweep step: backward (compute u[*,j]) ------------------
    //
    // For fixed j, and for all interior i:
    //
    //   u(j, i) = p(j, i) * u(j+1, i) + q(j, i);
    //
    Func row_backward("row_backward");
    {
        Expr u_new = P_img(j_param, i) * U_img(j_param + 1, i) +
                     Q_img(j_param, i);
        row_backward(i) = u_new;
    }

    // -------------------- Schedules for the step Funcs ---------------------------
    {
        Target target = get_host_target();
        int vec_width = target.natural_vector_size(type_of<num_t>());
        int tile = std::max(1, vec_width * 8);

        Var io("io"), ii("ii");

        // Compute each step Func at root, tiled over i to allow both
        // parallelism and vectorization. These are small 1D pipelines,
        // so we keep the schedule simple and robust.
        col_forward
            .compute_root()
            .split(i, io, ii, tile)
            .vectorize(ii, vec_width)
            .parallel(io);

        col_backward
            .compute_root()
            .split(i, io, ii, tile)
            .vectorize(ii, vec_width)
            .parallel(io);

        row_forward
            .compute_root()
            .split(i, io, ii, tile)
            .vectorize(ii, vec_width)
            .parallel(io);

        row_backward
            .compute_root()
            .split(i, io, ii, tile)
            .vectorize(ii, vec_width)
            .parallel(io);
    }

    // Bind the coefficient Params once per kernel_adi call.
    a_p.set(a_val);
    b_p.set(b_val);
    c_p.set(c_val);
    d_p.set(d_val);
    e_p.set(e_val);
    f_p.set(f_val);
    one_plus_2d_p.set(one_plus_2d_val);
    one_plus_2a_p.set(one_plus_2a_val);

    // Bind the ImageParams to the actual Buffers.
    U_img.set(u);
    V_img.set(v);
    P_img.set(p);
    Q_img.set(q);

    // 1D scratch buffers for a single sweep step over i (interior region).
    // We index them from 1 to n-2 to match the C loop indices and to keep
    // the interior region cleanly separated from the boundaries.
    Buffer<num_t> p_row(n - 2), q_row(n - 2), v_row(n - 2), u_row(n - 2);
    p_row.set_min(1);
    q_row.set_min(1);
    v_row.set_min(1);
    u_row.set_min(1);

    // -------------------------------------------------------------------------
    // Time-stepping loop
    // -------------------------------------------------------------------------
    for (int t = 1; t <= tsteps; t++) {

        // ========================================================
        // Column sweep (operates primarily along the "j" dimension)
        // ========================================================

        // Boundary/initial conditions for this sweep.
        for (int i_idx = 1; i_idx < n - 1; i_idx++) {
            v(i_idx, 0) = num_t(1.0);

            p(0, i_idx) = num_t(0.0);
            q(0, i_idx) = v(i_idx, 0);
        }

        // Forward sweep: compute p(j, i) and q(j, i) for j = 1..n-2.
        // We reformulate the original i-outer, j-inner loop as j-outer,
        // i-inner, which is legal because there are no dependencies
        // across i. For each fixed j, the Halide pipeline evaluates all i
        // in parallel/vectorized fashion.
        for (int j = 1; j < n - 1; j++) {
            j_param.set(j);

            Realization r_fwd({p_row, q_row});
            col_forward.realize(r_fwd);

            // Scatter the 1D results back into the 2D p and q Buffers.
            for (int i_idx = 1; i_idx < n - 1; i_idx++) {
                p(j, i_idx) = p_row(i_idx);
                q(j, i_idx) = q_row(i_idx);
            }
        }

        // Top boundary of v.
        for (int i_idx = 1; i_idx < n - 1; i_idx++) {
            v(i_idx, n - 1) = num_t(1.0);
        }

        // Back substitution along j: v(i, j) = p(j, i) * v(i, j+1) + q(j, i)
        // for j = n-2 .. 1. Again, we make j the outer loop and evaluate
        // all i in parallel via Halide.
        for (int j = n - 2; j >= 1; j--) {
            j_param.set(j);

            col_backward.realize(v_row);

            for (int i_idx = 1; i_idx < n - 1; i_idx++) {
                v(i_idx, j) = v_row(i_idx);
            }
        }

        // ========================================================
        // Row sweep (operates primarily along the "j" dimension)
        // ========================================================

        // Boundary/initial conditions for this sweep.
        for (int i_idx = 1; i_idx < n - 1; i_idx++) {
            u(0, i_idx) = num_t(1.0);

            p(0, i_idx) = num_t(0.0);
            q(0, i_idx) = u(0, i_idx);
        }

        // Forward sweep: compute p(j, i) and q(j, i) for j = 1..n-2,
        // now using the v array.
        for (int j = 1; j < n - 1; j++) {
            j_param.set(j);

            Realization r_fwd_row({p_row, q_row});
            row_forward.realize(r_fwd_row);

            for (int i_idx = 1; i_idx < n - 1; i_idx++) {
                p(j, i_idx) = p_row(i_idx);
                q(j, i_idx) = q_row(i_idx);
            }
        }

        // Right boundary of u.
        for (int i_idx = 1; i_idx < n - 1; i_idx++) {
            u(n - 1, i_idx) = num_t(1.0);
        }

        // Back substitution along j for u: u(j, i) = p(j, i) * u(j+1, i) + q(j, i)
        // for j = n-2 .. 1.
        for (int j = n - 2; j >= 1; j--) {
            j_param.set(j);

            row_backward.realize(u_row);

            for (int i_idx = 1; i_idx < n - 1; i_idx++) {
                u(j, i_idx) = u_row(i_idx);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Problem sizes
    int n      = N;
    int tsteps = TSTEPS;

    // Allocate Buffers.
    // C array A[i][j] is mapped as Buffer A(j, i) with shape (n, n).
    Halide::Buffer<num_t, 2> u(n, n);
    Halide::Buffer<num_t, 2> v(n, n);
    Halide::Buffer<num_t, 2> p(n, n);
    Halide::Buffer<num_t, 2> q(n, n);

    // Initialize u; v, p, q are scratch arrays.
    init_array(n, u);

    // Time only the kernel.
    auto start = std::chrono::high_resolution_clock::now();

    kernel_adi(tsteps, n, u, v, p, q);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally dump the result (u) to stderr, in row-major order.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // u[i][j] == u_buf(j, i)
                std::cerr << u(j, i) << '\n';
            }
        }
    }

    // Print timing (seconds) to stdout.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}