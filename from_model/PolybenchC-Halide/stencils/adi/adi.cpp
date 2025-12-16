#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

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
    // x == j (fast), y == i (slow)
    Var x("x"), y("y");
    Func init_u("init_u");

    // u[i][j] = (i + n - j) / n  ==>  u(j, i) = ...
    init_u(x, y) = cast<num_t>(y + n - x) / cast<num_t>(n);

    // Simple schedule: vectorize along x and parallelize along y.
    // This runs once, so scheduling is not critical but is essentially free.
    init_u.vectorize(x, 8).parallel(y);

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
 * Optimization strategy:
 *   - Keep the exact ADI algorithm and boundary conditions.
 *   - Hoist all scalar coefficients outside the time loop.
 *   - Replace the 2D scratch arrays p,q with 1D per-line buffers
 *     (p_line, q_line) that are reused for each column/row solve.
 *     This:
 *       * avoids O(n^2) scratch traffic on p,q,
 *       * keeps p_line/q_line in cache/registers,
 *       * improves data locality without changing u or v.
 *   - The recurrences along j remain strictly sequential, as required.
 */
static void kernel_adi(int tsteps,
                       int n,
                       Halide::Buffer<num_t, 2> u,
                       Halide::Buffer<num_t, 2> v,
                       Halide::Buffer<num_t, 2> p_buf,
                       Halide::Buffer<num_t, 2> q_buf) {
    // --------------------------------------------------------------------
    // Hoist scalar coefficients (identical to the original code).
    // --------------------------------------------------------------------
    const num_t DX   = num_t(1.0) / num_t(n);
    const num_t DY   = num_t(1.0) / num_t(n);
    const num_t DT   = num_t(1.0) / num_t(tsteps);
    const num_t B1   = num_t(2.0);
    const num_t B2   = num_t(1.0);
    const num_t mul1 = B1 * DT / (DX * DX);
    const num_t mul2 = B2 * DT / (DY * DY);

    const num_t a = -mul1 / num_t(2.0);
    const num_t b = num_t(1.0) + mul1;
    const num_t c = a;
    const num_t d = -mul2 / num_t(2.0);
    const num_t e = num_t(1.0) + mul2;
    const num_t f = d;

    // Precompute combinations used in the inner loops to reduce flops.
    const num_t one_plus_2d = num_t(1.0) + num_t(2.0) * d;
    const num_t one_plus_2a = num_t(1.0) + num_t(2.0) * a;

    // --------------------------------------------------------------------
    // 1D scratch arrays for the tridiagonal solves along j.
    //
    // In the original code, p and q are N x N arrays. However, for each
    // fixed line (column/row) i, only p[i][*] and q[i][*] are ever used.
    // We can thus use a single 1D buffer per line and reuse it:
    //
    //   - During the column sweep, p_line/q_line model p[i][j], q[i][j]
    //     for a fixed i.
    //   - During the row sweep, they similarly model p[i][j], q[i][j].
    //
    // This avoids O(n^2) scratch traffic on p,q and keeps these
    // temporaries in L1 cache / registers. The p_buf and q_buf
    // arguments are retained for API compatibility but are no longer
    // used.
    // --------------------------------------------------------------------
    std::vector<num_t> p_line(n);
    std::vector<num_t> q_line(n);

    // Time-stepping loop
    for (int t = 1; t <= tsteps; t++) {

        // ================================================================
        // Column sweep: solve independent tridiagonal systems along j
        // for each fixed "column" i (in the original C notation).
        //
        // This follows the original algorithm exactly, but uses p_line
        // and q_line instead of 2D p and q.
        // ================================================================
        for (int i = 1; i < n - 1; i++) {
            // v[0][i] = 1.0  ==>  v(i, 0)
            v(i, 0) = num_t(1.0);

            // p[i][0] = 0.0, q[i][0] = v[0][i]
            p_line[0] = num_t(0.0);
            q_line[0] = v(i, 0);

            // Forward sweep along j: j = 1 .. n-2
            for (int j = 1; j < n - 1; j++) {
                // denom = a * p[i][j-1] + b;
                const num_t denom = a * p_line[j - 1] + b;
                // p[i][j] = -c / denom;
                p_line[j] = -c / denom;

                // q[i][j] =
                //   (-d * u[j][i-1]
                //    + (1 + 2*d) * u[j][i]
                //    - f * u[j][i+1]
                //    - a * q[i][j-1]) / denom;
                //
                // Recall mapping: C u[j][i] => Buffer u(i, j).
                const num_t num =
                    -d * u(i - 1, j) +
                    one_plus_2d * u(i, j) -
                    f * u(i + 1, j) -
                    a * q_line[j - 1];

                q_line[j] = num / denom;
            }

            // v[n-1][i] = 1.0  ==>  v(i, n - 1)
            v(i, n - 1) = num_t(1.0);

            // Back substitution along j: j = n-2 .. 1
            for (int j = n - 2; j >= 1; j--) {
                // v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
                v(i, j) = p_line[j] * v(i, j + 1) + q_line[j];
            }
        }

        // ================================================================
        // Row sweep: solve independent tridiagonal systems along j
        // for each fixed "row" i (in the original C notation).
        //
        // Uses the updated v from the column sweep and updates u.
        // Again, we use p_line and q_line as 1D temporaries.
        // ================================================================
        for (int i = 1; i < n - 1; i++) {
            // u[i][0] = 1.0  ==>  u(0, i)
            u(0, i) = num_t(1.0);

            // p[i][0] = 0.0, q[i][0] = u[i][0]
            p_line[0] = num_t(0.0);
            q_line[0] = u(0, i);

            // Forward sweep along j: j = 1 .. n-2
            for (int j = 1; j < n - 1; j++) {
                // denom = d * p[i][j-1] + e;
                const num_t denom = d * p_line[j - 1] + e;
                // p[i][j] = -f / denom;
                p_line[j] = -f / denom;

                // q[i][j] =
                //   (-a * v[i-1][j]
                //    + (1 + 2*a) * v[i][j]
                //    - c * v[i+1][j]
                //    - d * q[i][j-1]) / denom;
                //
                // Mapping: C v[i][j] => Buffer v(j, i)
                const num_t num =
                    -a * v(j, i - 1) +
                    one_plus_2a * v(j, i) -
                    c * v(j, i + 1) -
                    d * q_line[j - 1];

                q_line[j] = num / denom;
            }

            // u[i][n-1] = 1.0  ==>  u(n - 1, i)
            u(n - 1, i) = num_t(1.0);

            // Back substitution along j: j = n-2 .. 1
            for (int j = n - 2; j >= 1; j--) {
                // u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
                u(j, i) = p_line[j] * u(j + 1, i) + q_line[j];
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