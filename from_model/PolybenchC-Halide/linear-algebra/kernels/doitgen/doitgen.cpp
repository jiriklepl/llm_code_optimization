#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "doitgen.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize arrays A and C4 with the same values as the original C code.
 *
 * Original C layout:
 *   A[NR][NQ][NP]  accessed as A[i][j][k]
 *   C4[NP][NP]     accessed as C4[i][j]
 *
 * Halide layout:
 *   A(p, q, r)  with Buffer size (NP, NQ, NR)
 *     mapping: A(p, q, r) == A_c[r][q][p]
 *
 *   C4(p, s) with Buffer size (NP, NP)
 *     mapping: C4(p, s) == C4_c[s][p]
 */
static void init_array(int nr, int nq, int np,
                       Halide::Buffer<num_t, 3> A,
                       Halide::Buffer<num_t, 2> C4) {
    Var p("p"), q("q"), r("r");
    Var i("i"), j("j");

    Func init_A("init_A"), init_C4("init_C4");

    // A[i][j][k] = ( (i*j + k) % np ) / np;
    // Mapping: i -> r, j -> q, k -> p  =>  A(p, q, r)
    init_A(p, q, r) =
        cast<num_t>((r * q + p) % np) / cast<int>(np);

    // C4[i][j] = ( (i*j) % np ) / np;
    // Mapping: original C4[s][p] -> C4(p, s)
    init_C4(i, j) =
        cast<num_t>((j * i) % np) / cast<int>(np);

    // Simple, but effective schedule for initialization:
    // - p is unit-stride in memory for both A and C4, so we vectorize p/i.
    // - r and j are outer dimensions, so we parallelize across them.
    Target target = get_jit_target_from_environment();
    int vec_width = Halide::natural_vector_size(type_of<num_t>(), target);
    if (vec_width < 1) vec_width = 1;

    init_A
        .reorder(p, q, r)       // r outermost, q middle, p innermost
        .vectorize(p, vec_width)
        .parallel(r);

    init_C4
        .reorder(i, j)          // j outer, i inner
        .vectorize(i, vec_width)
        .parallel(j);

    init_A.realize(A);
    init_C4.realize(C4);
}

int main(int argc, char *argv[]) {
    // Problem size (from doitgen.hpp / PolyBench)
    int nr = NR;
    int nq = NQ;
    int np = NP;

    // Buffers corresponding to:
    //   A[NR][NQ][NP] -> A(np, nq, nr)  (p, q, r)
    //   C4[NP][NP]    -> C4(np, np)
    Halide::Buffer<num_t, 3> A(np, nq, nr);
    Halide::Buffer<num_t, 2> C4(np, np);

    // Initialize data
    init_array(nr, nq, np, A, C4);

    // ImageParams for Halide pipeline
    ImageParam A_param(type_of<num_t>(), 3, "A_param");
    ImageParam C4_param(type_of<num_t>(), 2, "C4_param");

    A_param.set(A);
    C4_param.set(C4);

    // Vars and RDom
    Var p("p"), q("q"), r("r");
    // Sum over 's' corresponds to the inner-most loop over NP
    // Use the known problem size 'np' so Halide can better optimize.
    RDom s(0, np, "s");

    // Main computation:
    // For each (r, q, p):
    //   A_new[r][q][p] = sum_{s=0..NP-1} A_old[r][q][s] * C4[s][p]
    //
    // Mapping to Halide:
    //   A_old[r][q][s] -> A_param(s, q, r)
    //   C4[s][p]       -> C4_param(p, s)
    Func doitgen("doitgen");

    // Pure definition: initialize output to zero.
    doitgen(p, q, r) = cast<num_t>(0);

    // Reduction definition: accumulate contributions from all s.
    doitgen(p, q, r) += A_param(s, q, r) * C4_param(p, s);

    // -------------------------------
    // Optimized schedule for doitgen
    // -------------------------------
    //
    // High-level goals:
    //   * Reuse A_param(s, q, r) across all p for a given (r, q, s),
    //     so each A element is loaded once instead of NP times.
    //   * Make p the innermost loop everywhere for unit-stride, SIMD-
    //     friendly access to both C4 and the output.
    //   * Parallelize across r (and implicitly q) to exploit many cores.
    //   * Vectorize across p according to the natural SIMD width.
    //
    // Mathematical semantics:
    //   A naive implementation would be:
    //
    //     for r, q, p:
    //       acc = 0
    //       for s:
    //         acc += A_param(s, q, r) * C4_param(p, s);
    //       doitgen(p, q, r) = acc;
    //
    // We instead evaluate as:
    //
    //     for r, q:
    //       for p: doitgen(p, q, r) = 0
    //       for s:
    //         a = A_param(s, q, r);
    //         for p:
    //           doitgen(p, q, r) += a * C4_param(p, s);
    //
    // which is equivalent up to floating-point roundoff, but:
    //   - Loads each A_param(s,q,r) exactly once per (r,q).
    //   - Streams C4_param(p,s) along p (unit-stride) for each s.
    //   - Writes doitgen(p,q,r) in unit-stride order.
    //
    // This corresponds to moving the reduction loop 's' outside 'p',
    // while keeping it inside 'q' and 'r'.
    Target target = get_jit_target_from_environment();
    int vec_width = Halide::natural_vector_size(type_of<num_t>(), target);
    if (vec_width < 1) vec_width = 1;

    // Pure stage: zero initialization.
    // Loop order: r (outer), q, p (innermost). Vectorize along p and
    // parallelize across r.
    doitgen
        .reorder(p, q, r)          // r outermost, q middle, p innermost
        .vectorize(p, vec_width)   // SIMD across contiguous p-dimension
        .parallel(r);              // different r-slices run in parallel

    // Update stage (reduction over s):
    // Reorder to r -> q -> s -> p, so for each (r,q,s) we update all p.
    doitgen.update()
        .reorder(p, s, q, r)       // loops: r outer, q, s, p inner
        .vectorize(p, vec_width)   // SIMD across p
        .parallel(r);              // parallel over independent r-slices

    // Bind JIT and compile
    doitgen.compile_jit(target);

    // Measure execution time of the kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Realize into a new buffer A_out with same layout as A
    Halide::Buffer<num_t, 3> A_out(np, nq, nr);
    doitgen.realize(A_out);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result (similar spirit to PolyBench print_array)
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int r_idx = 0; r_idx < nr; r_idx++) {
            for (int q_idx = 0; q_idx < nq; q_idx++) {
                for (int p_idx = 0; p_idx < np; p_idx++) {
                    // Mapping back to A[r][q][p] -> A_out(p, q, r)
                    std::cerr << A_out(p_idx, q_idx, r_idx) << '\n';
                }
            }
        }
    }

    // Print kernel execution time in seconds, with 6 decimal places
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}