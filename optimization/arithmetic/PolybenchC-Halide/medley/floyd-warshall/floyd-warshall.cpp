#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

#ifndef N
#define N 1024
#endif

#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

using num_t = DATA_TYPE;
using namespace Halide;

static void init_array(int n, Halide::Buffer<num_t, 2> path) {
    Var i("i"), j("j");
    Func init("init");

    Expr ii = i, jj = j;
    Expr base = (ii * jj) % 7 + 1;
    Expr sum = ii + jj;
    Expr special = (sum % 13 == 0) || (sum % 7 == 0) || (sum % 11 == 0);

    // Note: Buffer is indexed as (x=j, y=i)
    init(j, i) = select(special, cast<num_t>(999), cast<num_t>(base));

    // Simple schedule to improve init throughput
    init.vectorize(j, 32).parallel(i);

    init.realize(path);
}

int main(int argc, char* argv[]) {
    int n = N;

    // Buffers: map path[i][j] -> path(j, i) to keep j as the fastest-varying dim (stride-1).
    Buffer<num_t, 2> path_curr(n, n);
    Buffer<num_t, 2> path_next(n, n);

    // Initialize input
    init_array(n, path_curr);

    // Define a single-k step of Floyd–Warshall:
    // path[i][j] = min(path[i][j], path[i][k] + path[k][j])
    // Buffer coordinates mapping:
    //   path(i,j) -> P(j,i); path(i,k) -> P(k,i); path(k,j) -> P(j,k)
    ImageParam P(type_of<num_t>(), 2, "P");
    Param<int> pk("k");

    Var i("i"), j("j");
    Func col("col"), row("row"), step("step");

    // Cache the k-th column and k-th row for this iteration to improve locality.
    // col(i) = P(k, i)  (depends only on i)
    // row(j) = P(j, k)  (depends only on j)
    col(i) = P(pk, i);
    row(j) = P(j, pk);

    // Main update
    step(j, i) = min(P(j, i), col(i) + row(j));

    // Schedule: vectorize across the stride-1 dimension (j), parallelize across rows (i),
    // and compute the 1D helpers at root for reuse.
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    // Compute helpers once per iteration over k
    col.compute_root().vectorize(i, 32);
    row.compute_root().vectorize(j, 32);

    // Tile for better cache locality; vectorize over contiguous x (j), parallelize tiles in y (i)
    step
        .tile(j, i, xo, yo, xi, yi, 128, 64)
        .vectorize(xi, 32)
        .parallel(yo);

    // Compile for the host target with any features requested via environment
    const Target target = get_jit_target_from_environment();
    step.compile_jit(target);

    auto start = std::chrono::high_resolution_clock::now();

    // Iterate k on the host with double-buffering
    for (int k = 0; k < n; k++) {
        pk.set(k);
        P.set(path_curr);
        step.realize(path_next);
        std::swap(path_curr, path_next);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Print results to stderr
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                std::cerr << path_curr(jj, ii) << '\n';
            }
        }
    }

    // Report runtime in seconds to stdout
    std::cout << std::fixed << std::setprecision(6) << duration.count() << std::endl;

    return 0;
}
