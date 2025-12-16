#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common PolyBench-style definitions (DATA_TYPE, SCALAR_VAL, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (e.g., N)
#include "durbin.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// Initialize the input array r[i] = (n + 1 - i)
static void init_array(int n, Halide::Buffer<num_t, 1> r) {
    Var i("i");
    Func init_r("init_r");

    // r[i] = n + 1 - i
    init_r(i) = cast<num_t>(n + 1 - i);

    init_r.realize(r);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Input and output arrays (1D)
    Buffer<num_t, 1> r(n);
    // We'll keep two buffers for y to implement double-buffering.
    Buffer<num_t, 1> y_prev(n);
    Buffer<num_t, 1> y_curr(n);

    // Initialize input
    init_array(n, r);

    // ----------------------------------------------------------------------
    // Halide setup for the Durbin kernel
    // ----------------------------------------------------------------------

    // ImageParams for the constant input r and the current y-vector
    ImageParam r_param(type_of<num_t>(), 1, "r_param");
    ImageParam y_param(type_of<num_t>(), 1, "y_param");

    // Parameters for the current iteration index k and scalar alpha
    Param<int>   k_param("k");
    Param<num_t> alpha_param("alpha");

    // Set the input parameter
    r_param.set(r);

    // ----------------------------------------------------------------------
    // Func to compute the inner sum:
    //
    //   sum = Σ_{i=0..k-1} r[k-i-1] * y[i]
    //
    // expressed as a reduction over i in [0, k_param)
    // ----------------------------------------------------------------------
    Func sum_func("sum");
    RDom ri(0, k_param, "ri");

    sum_func() = cast<num_t>(0);
    sum_func() += r_param(k_param - ri - 1) * y_param(ri);

    // ----------------------------------------------------------------------
    // Func to compute the updated prefix of y:
    //
    //   for i in 0..k-1:
    //     y_new[i] = y_old[i] + alpha * y_old[k-i-1]
    //
    // We only realize this Func over i = 0..k-1, so all accesses are in-bounds.
    // ----------------------------------------------------------------------
    Var i("i");
    Func update_prefix("update_prefix");
    update_prefix(i) = y_param(i) + alpha_param * y_param(k_param - i - 1);

    // Simple schedules: compute these Funcs at root
    sum_func.compute_root();
    update_prefix.compute_root();

    // JIT-compile the Halide code once, before timing
    sum_func.compile_jit();
    update_prefix.compile_jit();

    // ----------------------------------------------------------------------
    // Host-side Durbin recursion using the Halide Funcs for the inner work
    // ----------------------------------------------------------------------

    // Initial conditions:
    // y[0] = -r[0];
    // beta = 1.0;
    // alpha = -r[0];
    num_t alpha = -r(0);
    num_t beta  = (num_t)1.0;

    // Initialize y_prev
    y_prev(0) = -r(0);
    for (int idx = 1; idx < n; ++idx) {
        y_prev(idx) = (num_t)0;
    }

    // Attach the constant input r once
    r_param.set(r);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Main Durbin loop:
    //
    // for (k = 1; k < n; k++) {
    //   beta = (1 - alpha*alpha) * beta;
    //   sum  = Σ_{i=0..k-1} r[k-i-1]*y[i];
    //   alpha = -(r[k] + sum) / beta;
    //   for (i=0; i<k; i++) y[i] = y[i] + alpha*y[k-i-1];
    //   y[k] = alpha;
    // }
    //
    // We implement sum and the y-prefix update with Halide Funcs,
    // and manage the per-iteration alpha/beta and buffer swapping on the host.
    for (int k = 1; k < n; ++k) {
        // Update beta using the previous alpha
        beta = ((num_t)1.0 - alpha * alpha) * beta;

        // Bind k and current y_prev into the Halide pipeline
        k_param.set(k);
        y_param.set(y_prev);

        // Compute sum = Σ_{i=0..k-1} r[k-i-1]*y_prev[i]
        Buffer<num_t> sum_buf = sum_func.realize();
        num_t sum = sum_buf();

        // Compute the new alpha for this k
        alpha = -(r(k) + sum) / beta;

        // Bind the new alpha for the prefix-update Func
        alpha_param.set(alpha);

        // Compute y_curr[0..k-1] = y_prev[i] + alpha * y_prev[k-i-1]
        // We realize only the prefix [0, k), via a cropped view of y_curr.
        Buffer<num_t, 1> y_prefix = y_curr.cropped(0, 0, k);
        update_prefix.realize(y_prefix);

        // Set y_curr[k] = alpha
        y_curr(k) = alpha;

        // Swap y_prev and y_curr for the next iteration
        std::swap(y_prev, y_curr);
    }

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // ----------------------------------------------------------------------
    // Print result vector y (stored in y_prev) to stderr, one value per line,
    // mirroring the PolyBench print_array behavior.
    // ----------------------------------------------------------------------
    std::cerr << std::fixed << std::setprecision(2);
    for (int idx = 0; idx < n; ++idx) {
        std::cerr << y_prev(idx) << '\n';
    }

    // Print execution time (in seconds) to stdout
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}