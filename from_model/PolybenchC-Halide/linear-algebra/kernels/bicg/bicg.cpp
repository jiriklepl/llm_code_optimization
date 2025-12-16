#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, N, M, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "bicg.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// Initialize arrays using Halide to mirror the C init_array().
static void init_array(int m,
                       int n,
                       Halide::Buffer<num_t, 2> A,  // shape: (m, n) => A(j, i) == A[i][j] in C
                       Halide::Buffer<num_t, 1> r,  // length n
                       Halide::Buffer<num_t, 1> p)  // length m
{
    Var i("i"), j("j");

    Func init_p("init_p");
    Func init_r("init_r");
    Func init_A("init_A");

    // p[i] = (DATA_TYPE)(i % m) / m;
    init_p(i) = cast<num_t>(i % m) / cast<int>(m);

    // r[i] = (DATA_TYPE)(i % n) / n;
    init_r(i) = cast<num_t>(i % n) / cast<int>(n);

    // A[i][j] = (DATA_TYPE)(i * (j+1) % n) / n;
    // C layout: A[i][j] -> Halide: A(j, i)
    init_A(j, i) = cast<num_t>((i * (j + 1)) % n) / cast<int>(n);

    // A simple but reasonably efficient schedule for initialization.
    // This is not performance-critical compared to the main kernel,
    // but we still vectorize along the unit-stride dimension.
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // p and r are 1D and contiguous.
    init_p.compute_root().vectorize(i, vec_width);
    init_r.compute_root().vectorize(i, vec_width);

    // A is stored row-major in j (dimension 0), so vectorize in j.
    init_A.compute_root().vectorize(j, vec_width);

    init_p.realize(p);
    init_r.realize(r);
    init_A.realize(A);
}

int main(int argc, char *argv[]) {
    // Problem size (from bicg.hpp / PolyBench)
    int n = N;  // number of rows of A, and length of r, q
    int m = M;  // number of columns of A, and length of p, s

    // Buffers corresponding to the C arrays.
    //
    // C: DATA_TYPE A[N][M]  -> Halide: Buffer<num_t,2> A(m, n)  with A(j, i) == A[i][j]
    // C: DATA_TYPE s[M]     -> Halide: Buffer<num_t,1> s(m)
    // C: DATA_TYPE q[N]     -> Halide: Buffer<num_t,1> q(n)
    // C: DATA_TYPE p[M]     -> Halide: Buffer<num_t,1> p(m)
    // C: DATA_TYPE r[N]     -> Halide: Buffer<num_t,1> r(n)
    Halide::Buffer<num_t, 2> A(m, n);
    Halide::Buffer<num_t, 1> s(m);
    Halide::Buffer<num_t, 1> q(n);
    Halide::Buffer<num_t, 1> p(m);
    Halide::Buffer<num_t, 1> r(n);

    // Initialize data (equivalent to init_array in the C version).
    init_array(m, n, A, r, p);

    // Wrap input arrays in ImageParams for Halide.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");  // (m, n), accessed as A_param(j, i)
    ImageParam p_param(type_of<num_t>(), 1, "p_param");  // (m)
    ImageParam r_param(type_of<num_t>(), 1, "r_param");  // (n)

    A_param.set(A);
    p_param.set(p);
    r_param.set(r);

    // Halide variables.
    Var i("i"), j("j");

    // RDom extents derived from the input buffers (fixed for this problem size).
    RDom ri(0, r.dim(0).extent(), "ri");  // 0 .. n-1  (row index i)
    RDom rj(0, p.dim(0).extent(), "rj");  // 0 .. m-1  (column index j)

    // ------------------------------------------------------------------------
    // Halide formulation of the BiCG kernel.
    //
    // Original C kernel:
    //
    // for (i = 0; i < m; i++)
    //   s[i] = 0;
    // for (i = 0; i < n; i++) {
    //   q[i] = 0;
    //   for (j = 0; j < m; j++) {
    //     s[j] = s[j] + r[i] * A[i][j];
    //     q[i] = q[i] + A[i][j] * p[j];
    //   }
    // }
    //
    // Which is mathematically:
    //   s[j] = sum_{i=0..n-1} r[i] * A[i][j]   (A^T * r)
    //   q[i] = sum_{j=0..m-1} A[i][j] * p[j]   (A * p)
    // ------------------------------------------------------------------------

    // s[j] = sum_i r[i] * A[i][j]
    // Halide: A(i,j) => A_param(j, i)
    Func s_func("s");
    s_func(j) = cast<num_t>(0);
    s_func(j) += r_param(ri) * A_param(j, ri);

    // q[i] = sum_j A[i][j] * p[j]
    Func q_func("q");
    q_func(i) = cast<num_t>(0);
    q_func(i) += A_param(rj, i) * p_param(rj);

    // ------------------------------------------------------------------------
    // Optimized schedule
    //
    // Goals:
    //   - Good data locality for A:
    //       * For s: iterate over A row-wise (i outer, j inner) so that
    //         j (dimension 0) is unit stride.
    //       * For q: default order already streams A row-wise.
    //   - Exploit SIMD where memory access is unit-stride.
    //   - Use multicore parallelism where it is race-free and cheap.
    //
    // We keep the two reductions separate (one for s, one for q) for
    // simplicity and robustness, but apply scheduling to each.
    // ------------------------------------------------------------------------

    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // --- Schedule for s_func: s[j] = sum_i r[i] * A[i][j] -------------------
    //
    // The update definition is:
    //   for j:
    //     for ri:
    //       s[j] += r[ri] * A(j, ri);
    //
    // This traverses A with j outer and ri inner, which is strided in memory
    // because A is row-major in j (dimension 0). We reorder the loops so that
    // j is innermost and then vectorize over j to get unit-stride SIMD loads.
    s_func.compute_root();

    // Reorder update loops to: for ri (rows) outer, for j (cols) inner.
    // This makes the inner loop walk A(j, ri) with j contiguous.
    s_func.update()
        .reorder(j, ri)        // innermost j, then ri
        .vectorize(j, vec_width);  // SIMD across contiguous columns of A and s

    // We deliberately do not parallelize this reduction: parallelizing over
    // the reduction dimension (ri) is non-trivial without rfactor, and
    // parallelizing over j here would introduce nested parallelism with
    // limited benefit. The main parallel speedup comes from q_func.

    // --- Schedule for q_func: q[i] = sum_j A[i][j] * p[j] -------------------
    //
    // The update definition is:
    //   for i:
    //     for rj:
    //       q[i] += A(rj, i) * p[rj];
    //
    // Here rj iterates over columns j; since A is stored as A(j, i) with j
    // in dimension 0, rj is already the unit-stride dimension, so we keep
    // this order. Rows i are independent, so we parallelize across tiles of i.
    q_func.compute_root();

    Var io("io"), ii("ii");
    const int tile_i = 64;  // reasonable row-tile size for cache and parallelism

    q_func.update()
        .split(i, io, ii, tile_i)  // i = io * tile_i + ii
        .parallel(io);             // parallelize across row tiles

    // JIT-compile both functions once, so timing measures only execution.
    s_func.compile_jit(target);
    q_func.compile_jit(target);

    // Measure execution time of the "kernel" (computing both s and q).
    auto start = std::chrono::high_resolution_clock::now();

    s_func.realize(s);
    q_func.realize(q);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optional: print results, similar to print_array in the C version.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);

        // Print s (length m), newline every 20 elements.
        std::cerr << "s:";
        for (int idx = 0; idx < m; idx++) {
            if (idx % 20 == 0) {
                std::cerr << '\n';
            }
            std::cerr << s(idx) << ' ';
        }
        std::cerr << "\nq:";

        // Print q (length n), newline every 20 elements.
        for (int idx = 0; idx < n; idx++) {
            if (idx % 20 == 0) {
                std::cerr << '\n';
            }
            std::cerr << q(idx) << ' ';
        }
        std::cerr << '\n';
    }

    // Print elapsed time in seconds (like gemm.cpp).
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}