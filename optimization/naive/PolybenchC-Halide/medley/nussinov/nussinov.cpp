#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <string>

#include "Halide.h"

// include common definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (e.g. N, DATA_TYPE)
#include "nussinov.hpp"

using num_t  = DATA_TYPE;   // type used for table entries
using base_t = uint8_t;     // type used for RNA bases (0..3)

using namespace Halide;

// Initialize seq and table as in the original C code.
static void init_array(int n,
                       Halide::Buffer<base_t, 1> seq,
                       Halide::Buffer<num_t, 2> table) {
    Var i("i"), j("j");

    Func init_seq("init_seq"), init_table("init_table");

    // base is AGCT/0..3
    init_seq(i) = cast<base_t>((i + 1) % 4);

    // table[i][j] = 0 for all i,j
    // Remember: C uses table[i][j]; we map that to table(j, i).
    init_table(j, i) = cast<num_t>(0);

    // Simple but effective schedules for initialization:
    //  - vectorize across the fastest‑varying dimension
    //  - parallelize across the outer dimension for the 2D table
    init_seq
        .vectorize(i, 16);

    init_table
        .vectorize(j, 16)
        .parallel(i);

    init_seq.realize(seq);
    init_table.realize(table);
}

int main(int argc, char *argv[]) {
    // Problem size (from nussinov.hpp / PolyBench)
    int n = N;

    // Buffers:
    // seq[i]      -> seq(i)
    // table[i][j] -> table(j, i)  (C is row-major; Halide is column-major)
    Halide::Buffer<base_t, 1> seq(n);
    Halide::Buffer<num_t, 2> table(n, n);

    // Initialize data
    init_array(n, seq, table);

    // ImageParams to expose the current state to Halide
    ImageParam seq_param(type_of<base_t>(), 1, "seq_param");
    ImageParam table_param(type_of<num_t>(), 2, "table_param");

    // Scalar parameters for the DP indices (i, j)
    Param<int> i_param("i");
    Param<int> j_param("j");

    // Bind the buffers once; we mutate 'table' in-place on the host,
    // and Halide will always read the up-to-date values from it.
    seq_param.set(seq);
    table_param.set(table);

    // Define a Func that computes table[i][j] given the current
    // contents of 'table' and 'seq', for fixed (i, j).
    //
    // We will *not* attempt to express the full dynamic program in
    // Halide, because the recurrence has loop-carried dependencies
    // along j and k. Instead, we define a single-cell update as a
    // pure Halide pipeline, and drive the i/j loops from C++,
    // updating the table buffer in-place in the same order as the
    // original code.
    Func cell("cell");

    {
        Expr i = i_param;
        Expr j = j_param;

        // Current value table[i][j]
        // (C: table[i][j] -> Halide: table_param(j, i))
        Expr best = table_param(j, i);

        // if (j-1 >= 0) table[i][j] = max(table[i][j], table[i][j-1]);
        // In the original loops j runs from i+1..n-1, so j >= 1 and this
        // condition is always true inside the kernel. We can drop the check.
        best = max(best, table_param(j - 1, i));

        // if (i+1 < N) table[i][j] = max(table[i][j], table[i+1][j]);
        // In the original loops, any i for which j-loop runs satisfies i <= N-2,
        // so i+1 < N always holds. Drop the check.
        best = max(best, table_param(j, i + 1));

        // Diagonal / pairing term:
        //
        // if (j-1 >= 0 && i+1 < N) {
        //   if (i < j-1)
        //     table[i][j] = max(table[i][j],
        //                        table[i+1][j-1] + match(seq[i], seq[j]));
        //   else
        //     table[i][j] = max(table[i][j], table[i+1][j-1]);
        // }
        //
        // As above, the outer condition is always true in the original loop
        // nest. The only remaining condition is (i < j-1), which we keep.
        Expr b1 = cast<int32_t>(seq_param(i));
        Expr b2 = cast<int32_t>(seq_param(j));
        // match(b1,b2) = ((b1 + b2) == 3 ? 1 : 0)
        Expr match_val = select(b1 + b2 == 3, 1, 0);

        // Only allow bonding when i < j-1; otherwise, don't add match().
        Expr match_term = select(i < j - 1,
                                 cast<num_t>(match_val),
                                 cast<num_t>(0));

        // table[i+1][j-1] -> table_param(j-1, i+1)
        Expr diag_candidate = table_param(j - 1, i + 1) + match_term;
        best = max(best, diag_candidate);

        // Now the split term:
        //
        // for (k = i+1; k < j; k++)
        //   table[i][j] = max(table[i][j], table[i][k] + table[k+1][j]);
        //
        // Express this as a reduction over k in [i+1, j-1].
        Expr extent = j - i - 1; // >= 0 because j >= i+1 in the DP.
        RDom r(i + 1, extent, "r");

        // Pure definition: starting value for the reduction.
        cell() = best;

        // Reduction update: take the maximum over k.
        cell() = max(cell(),
                     table_param(r.x, i) + table_param(j, r.x + 1));

        // Schedule for the reduction:
        //
        // We keep the reduction serial but hint that the k‑loop should
        // be vectorized where profitable. The load from table_param(r.x, i)
        // is unit‑stride in memory (x dimension), so vectorization
        // improves locality for half of the accesses; the other term
        // is strided but still benefits from SIMD on many CPUs.
        Var dummy;   // no pure Vars, but we can still schedule the update
        RVar rxo("rxo"), rxi("rxi");
        cell.update()
            .split(r.x, rxo, rxi, 8)
            .vectorize(rxi);
    }

    // JIT-compile the per-cell pipeline once.
    Target target = get_host_target();
    cell.compile_jit(target);

    // Reuse a single 0‑dimensional buffer for the cell() result to
    // avoid repeated allocations inside the DP loops.
    Halide::Buffer<num_t> cell_buf = Halide::Buffer<num_t>::scalar();

    // Time the DP kernel (the i/j loops plus Halide calls).
    auto start = std::chrono::high_resolution_clock::now();

    // Original kernel:
    //
    // for (i = N-1; i >= 0; i--) {
    //   for (j = i+1; j < N; j++) {
    //     ... updates to table[i][j] ...
    //   }
    // }
    //
    // We reproduce this order exactly on the host, but each update to
    // table[i][j] is done by calling the Halide Func 'cell'.
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            i_param.set(i);
            j_param.set(j);

            // cell() is a zero-dimensional Func. Realize into the
            // preallocated scalar buffer to avoid per-call allocations.
            cell.realize(cell_buf);
            num_t new_val = cell_buf(); // zero-dimensional Buffer: operator() with 0 args

            // Write back to our DP table.
            table(j, i) = new_val;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the live-out data (upper triangle of the table),
    // similar to the PolyBench print_array() function.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        int t = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (t % 20 == 0) {
                    std::cerr << '\n';
                }
                std::cerr << table(j, i) << ' ';
                ++t;
            }
        }
        std::cerr << '\n';
    }

    // Print kernel execution time (in seconds, to 6 decimal places).
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}