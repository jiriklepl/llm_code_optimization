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
    init_p(i) = cast<num_t>(i % m) / Expr(m);

    // r[i] = (DATA_TYPE)(i % n) / n;
    init_r(i) = cast<num_t>(i % n) / Expr(n);

    // A[i][j] = (DATA_TYPE)(i * (j+1) % n) / n;
    // C layout: A[i][j] -> Halide: A(j, i)
    init_A(j, i) = cast<num_t>((i * (j + 1)) % n) / Expr(n);

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
    ImageParam A_param(type_of<num_t>(), 2, "A_param");  // (m, n)
    ImageParam p_param(type_of<num_t>(), 1, "p_param");  // (m)
    ImageParam r_param(type_of<num_t>(), 1, "r_param");  // (n)

    A_param.set(A);
    p_param.set(p);
    r_param.set(r);

    // Halide variables.
    Var i("i"), j("j");

    // Additional Vars for scheduling (tiling, parallelism, vectorization).
    Var jo("jo"), ji("ji");
    Var io("io"), ii("ii");

    // RDom extents derived from the input buffers (fixed for this problem size).
    RDom ri(0, r.dim(0).extent(), "ri");  // 0 .. n-1
    RDom rj(0, p.dim(0).extent(), "rj");  // 0 .. m-1

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
    // Optimized CPU schedule
    //
    // Goals:
    //  - Parallelize across independent output indices (j for s, i for q)
    //  - Vectorize across output indices where beneficial
    //  - For s(j), restructure loops so that we iterate A(j, i) with j as
    //    the innermost, vectorized dimension to improve spatial locality
    //    in A and in s.
    //
    // Note: We keep the two GEMV operations mathematically identical to
    // the original formulation; we only change loop ordering and tiling.
    // ------------------------------------------------------------------------

    // Reasonable defaults; 8-wide vectorization works well for typical
    // 32-bit types on x86 with AVX2, but will gracefully degrade if the
    // hardware has a narrower native vector width.
    const int vector_width = 8;
    const int tile_j = 64;  // tile size over j for s
    const int tile_i = 64;  // tile size over i for q

    // ---- Schedule for s_func: s[j] = sum_i r[i] * A[j,i] ----
    //
    // We:
    //  - compute_root so s is materialized as a 1D array
    //  - tile j into (jo, ji)
    //  - parallelize across jo (independent chunks of s)
    //  - vectorize ji to compute several s[j] in SIMD
    //  - in the update step, reorder the reduction so that for each tile
    //    we loop over ri (rows of A) outside and ji (columns) inside,
    //    giving us contiguous access along A's first dimension and s.
    s_func.compute_root();
    s_func
        .split(j, jo, ji, tile_j)
        .parallel(jo)
        .vectorize(ji, vector_width);

    {
        // Work on the update stage of s_func.
        auto s_update = s_func.update();

        // First, make j the inner loop and ri the outer:
        //   before: for j: for ri
        //   after:  for ri: for j
        s_update
            .reorder(j, ri)
            // Then tile j into (jo, ji); loops become:
            //   for ri:
            //     for jo:
            //       for ji:
            .split(j, jo, ji, tile_j)
            // Finally reorder to:
            //   for jo (outer, parallel):
            //     for ri:
            //       for ji (innermost, vectorized)
            .reorder(ji, ri, jo)
            .parallel(jo)
            .vectorize(ji, vector_width);
    }

    // ---- Schedule for q_func: q[i] = sum_j A[j,i] * p[j] ----
    //
    // For q, each output q[i] is independent, so we:
    //  - compute_root
    //  - tile i into (io, ii)
    //  - parallelize across io (chunks of rows)
    //  - vectorize across ii to compute several q[i] at once
    //  - keep the reduction over rj innermost to stream through A(j,i)
    //    row-wise and reuse p[j].
    q_func.compute_root();
    q_func
        .split(i, io, ii, tile_i)
        .parallel(io)
        .vectorize(ii, vector_width);

    {
        auto q_update = q_func.update();
        q_update
            .split(i, io, ii, tile_i)
            .parallel(io)
            .vectorize(ii, vector_width);
    }

    // JIT-compile both functions once, so timing measures only execution.
    Target target = get_jit_target_from_environment();
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