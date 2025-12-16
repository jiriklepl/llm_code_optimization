#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdint>

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

    // --------------------------------------------------------------------
    // Per-cell Halide pipeline:
    //
    // Given the current contents of 'table' and 'seq', and scalar
    // indices (i_param, j_param), compute table[i][j] according to
    // the Nussinov recurrence.
    //
    // We cannot express the full dynamic program as a single Halide
    // Func (it has loop-carried dependencies along both i and j),
    // so we follow the recommended pattern for DP:
    //
    //   - Define a *single-cell* transform as a pure Halide pipeline.
    //   - Drive the outer DP loops from C++, calling the pipeline
    //     for each (i, j) in a dependency-respecting order.
    //
    // This preserves correctness while allowing Halide to optimize
    // the innermost (O(n)) split reduction for each cell.
    // --------------------------------------------------------------------

    // Reduction domain for the split term:
    //   for (k = i+1; k < j; k++)
    //     best = max(best, table[i][k] + table[k+1][j]);
    //
    // We parameterize the reduction bounds by i_param and j_param, so
    // the same compiled kernel works for all (i,j) pairs; each call
    // just changes the scalar Params.
    RDom r(i_param + 1, j_param - i_param - 1, "r");

    Func cell("cell");
    {
        Expr i = i_param;
        Expr j = j_param;

        // Start from the current value table[i][j].
        // In this benchmark table[i][j] is 0 before any updates, but
        // we keep the load to match the original C semantics exactly.
        Expr best = table_param(j, i);

        // Case 1: j is unpaired with anything > j-1:
        //   best = max(best, table[i][j-1]);
        Expr from_left = table_param(j - 1, i);
        best = max(best, from_left);

        // Case 2: i is unpaired:
        //   best = max(best, table[i+1][j]);
        Expr from_down = table_param(j, i + 1);
        best = max(best, from_down);

        // Case 3: pairing / diagonal term.
        //
        // match(seq[i], seq[j]) = 1 if bases complement, else 0.
        // PolyBench Nussinov uses (b1 + b2 == 3 ? 1 : 0).
        Expr b1        = cast<int32_t>(seq_param(i));
        Expr b2        = cast<int32_t>(seq_param(j));
        Expr match_val = select(b1 + b2 == 3, 1, 0); // Int(32)

        // Pairing only allowed when there is at least one base
        // between i and j, i.e. i < j-1.
        Expr pair_allowed = i < (j - 1);

        // Branchless: either 0 or 1 in num_t, depending on pair_allowed.
        Expr match_term =
            cast<num_t>(match_val) * cast<num_t>(pair_allowed);

        // table[i+1][j-1] -> table_param(j-1, i+1)
        Expr diag_base      = table_param(j - 1, i + 1);
        Expr diag_candidate = diag_base + match_term;
        best = max(best, diag_candidate);

        // Split term:
        //
        // for (k = i+1; k < j; k++)
        //   best = max(best, table[i][k] + table[k+1][j]);
        //
        // Implemented as a reduction over r.x in [i+1, j-1].
        cell() = best;
        cell() = max(cell(),
                     table_param(r.x, i) + table_param(j, r.x + 1));
    }

    // --------------------------------------------------------------------
    // Schedule for the per-cell pipeline
    // --------------------------------------------------------------------
    //
    // The Func itself is scalar (zero-dimensional); the only sizable
    // loop is the reduction over r.x. We keep the Func computed at
    // root, and partially unroll the reduction to reduce loop overhead
    // and expose a bit of instruction-level parallelism in the inner
    // k-loop.
    cell.compute_root();
    {
        RVar r_outer("r_outer"), r_inner("r_inner");
        cell.update()
            .split(r.x, r_outer, r_inner, 4)  // r.x = 4 * r_outer + r_inner
            .unroll(r_inner);                 // fully unroll the 4‑element inner loop
    }

    // JIT-compile the per-cell pipeline once.
    Target target = get_jit_target_from_environment();
    cell.compile_jit(target);

    // --------------------------------------------------------------------
    // Host-side dynamic program
    //
    // Original C kernel:
    //
    //   for (i = N-1; i >= 0; i--)
    //     for (j = i+1; j < N; j++)
    //       ... recurrence for table[i][j] ...
    //
    // All dependencies of table[i][j] refer only to entries with
    // strictly smaller span (j - i), so we can equivalently evaluate
    // the DP in *diagonal* (wavefront) order:
    //
    //   for (len = 1; len < N; len++)        // span = j - i
    //     for (i = 0; i + len < N; i++) {
    //       j = i + len;
    //       ... same recurrence ...
    //     }
    //
    // This ordering:
    //   - Preserves correctness (all dependencies live on smaller spans)
    //   - Improves temporal locality compared to the descending-i order
    //   - Still leaves each (i,j) update independent within a fixed len
    //
    // We keep the DP itself on the host, but each cell update is
    // delegated to the compiled Halide pipeline.
    // --------------------------------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    // Reuse a single 1D buffer as the scalar output for cell().
    // A zero-dimensional Func can be realized into any Buffer whose
    // dimensionality is >= 0 and whose extra dimensions have extent 1.
    Halide::Buffer<num_t> cell_buf(1);

    for (int len = 1; len < n; ++len) {
        // For this subsequence length, i ranges so that j = i + len is < n.
        for (int i = 0; i < n - len; ++i) {
            int j = i + len;

            // Bind the current DP indices.
            i_param.set(i);
            j_param.set(j);

            // Evaluate the per-cell pipeline.
            cell.realize(cell_buf);
            num_t new_val = cell_buf(0);

            // Write back to our DP table (remember the j,i -> [i][j] mapping).
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