#include <chrono>
#include <iomanip>
#include <iostream>

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
 */
static void kernel_adi(int tsteps,
                       int n,
                       Halide::Buffer<num_t, 2> u,
                       Halide::Buffer<num_t, 2> v,
                       Halide::Buffer<num_t, 2> p,
                       Halide::Buffer<num_t, 2> q) {
    // Scalar parameters
    num_t DX   = num_t(1.0) / num_t(n);
    num_t DY   = num_t(1.0) / num_t(n);
    num_t DT   = num_t(1.0) / num_t(tsteps);
    num_t B1   = num_t(2.0);
    num_t B2   = num_t(1.0);
    num_t mul1 = B1 * DT / (DX * DX);
    num_t mul2 = B2 * DT / (DY * DY);

    num_t a = -mul1 / num_t(2.0);
    num_t b = num_t(1.0) + mul1;
    num_t c = a;
    num_t d = -mul2 / num_t(2.0);
    num_t e = num_t(1.0) + mul2;
    num_t f = d;

    // Time-stepping loop
    for (int t = 1; t <= tsteps; t++) {

        // ------------------------------------------------------------
        // Column sweep
        // ------------------------------------------------------------
        for (int i = 1; i < n - 1; i++) {
            // v[0][i] = 1.0;
            v(i, 0) = num_t(1.0);

            // p[i][0] = 0.0;
            p(0, i) = num_t(0.0);

            // q[i][0] = v[0][i];
            q(0, i) = v(i, 0);

            // Forward sweep along j
            for (int j = 1; j < n - 1; j++) {
                // p[i][j] = -c / (a * p[i][j-1] + b);
                num_t denom = a * p(j - 1, i) + b;
                p(j, i) = -c / denom;

                // q[i][j] =
                //   (-d * u[j][i-1]
                //    + (1 + 2*d) * u[j][i]
                //    - f * u[j][i+1]
                //    - a * q[i][j-1]) /
                //   (a * p[i][j-1] + b);
                num_t num = -d * u(i - 1, j)
                            + (num_t(1.0) + num_t(2.0) * d) * u(i, j)
                            - f * u(i + 1, j)
                            - a * q(j - 1, i);
                q(j, i) = num / denom;
            }

            // v[n-1][i] = 1.0;
            v(i, n - 1) = num_t(1.0);

            // Back substitution along j (descending)
            for (int j = n - 2; j >= 1; j--) {
                // v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
                v(i, j) = p(j, i) * v(i, j + 1) + q(j, i);
            }
        }

        // ------------------------------------------------------------
        // Row sweep
        // ------------------------------------------------------------
        for (int i = 1; i < n - 1; i++) {
            // u[i][0] = 1.0;
            u(0, i) = num_t(1.0);

            // p[i][0] = 0.0;
            p(0, i) = num_t(0.0);

            // q[i][0] = u[i][0];
            q(0, i) = u(0, i);

            // Forward sweep along j
            for (int j = 1; j < n - 1; j++) {
                // p[i][j] = -f / (d * p[i][j-1] + e);
                num_t denom = d * p(j - 1, i) + e;
                p(j, i) = -f / denom;

                // q[i][j] =
                //   (-a * v[i-1][j]
                //    + (1 + 2*a) * v[i][j]
                //    - c * v[i+1][j]
                //    - d * q[i][j-1]) /
                //   (d * p[i][j-1] + e);
                num_t num = -a * v(j, i - 1)
                            + (num_t(1.0) + num_t(2.0) * a) * v(j, i)
                            - c * v(j, i + 1)
                            - d * q(j - 1, i);
                q(j, i) = num / denom;
            }

            // u[i][n-1] = 1.0;
            u(n - 1, i) = num_t(1.0);

            // Back substitution along j (descending)
            for (int j = n - 2; j >= 1; j--) {
                // u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
                u(j, i) = p(j, i) * u(j + 1, i) + q(j, i);
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