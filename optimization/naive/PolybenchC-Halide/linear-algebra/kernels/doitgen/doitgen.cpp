#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

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
    // Mapping: i -> j (second index), j -> i (first index) => C4(first=j, second=i)
    init_C4(i, j) =
        cast<num_t>((j * i) % np) / cast<int>(np);

    // -------------  Scheduling for initialization -------------
    // Vectorize along the innermost (contiguous) dimension and
    // parallelize outer dimensions to speed up initialization.
    Target init_target = get_host_target();
    int vec_width = init_target.natural_vector_size(type_of<num_t>());
    vec_width = std::max(1, std::min(vec_width, np));

    // For A: p is innermost, r is outer – good for parallelizing across r.
    init_A
        .reorder(p, q, r)   // p innermost
        .vectorize(p, vec_width)
        .parallel(r);

    // For C4: i is innermost, j outer – parallelize rows, vectorize columns.
    init_C4
        .reorder(i, j)
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
    RDom s(0, A.dim(0).extent(), "s");  // A.dim(0).extent() == np

    // Main computation:
    // For each (r, q, p):
    //   A_new[r][q][p] = sum_{s=0..NP-1} A_old[r][q][s] * C4[s][p]
    //
    // Mapping to Halide:
    //   A_old[r][q][s] -> A_param(s, q, r)
    //   C4[s][p]       -> C4_param(p, s)
    Func doitgen("doitgen");

    doitgen(p, q, r) = cast<num_t>(0);
    doitgen(p, q, r) += A_param(s, q, r) * C4_param(p, s);

    // -------------  Scheduling for the main kernel -------------
    // We want:
    //   * p as innermost dimension (contiguous) for good SIMD.
    //   * Parallel work across (r, q) tiles.
    //   * Reduction over s inside each (r, q, p), keeping s scalar
    //     and p vectorized for good memory access on C4(p, s).
    //
    // The reduction is associative and independent across (r, q, p),
    // so it is safe to parallelize over r and q.
    Target target = get_host_target();
    int vec_width = target.natural_vector_size(type_of<num_t>());
    vec_width = std::max(1, std::min(vec_width, np));

    Var rq("rq");

    // Compute the whole 3D output at root.
    doitgen.compute_root();

    // Pure definition: arrange loops as rq (fused r,q) outermost, p innermost
    // and vectorized; parallelize over rq.
    doitgen
        .reorder(p, q, r)      // innermost -> outermost: p, q, r
        .fuse(r, q, rq)        // fuse r and q into a single parallel dimension
        .parallel(rq)
        .vectorize(p, vec_width);

    // Reduction update step: same loop structure, but also place the
    // reduction variable 's' just outside the vectorized p loop:
    //   for rq (parallel)
    //     for s
    //       for p (vectorized)
    doitgen.update()
        .reorder(p, s, q, r)   // innermost -> outermost: p, s, q, r
        .fuse(r, q, rq)
        .parallel(rq)
        .vectorize(p, vec_width);

    // Bind JIT and compile for the host target
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