#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "lu.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

// -----------------------------------------------------------------------------
// Array initialization: translate Polybench init_array() into a Halide pipeline
// with an optimized schedule for the SPD build B = A0 * A0^T.
// -----------------------------------------------------------------------------
static void init_array(int n, Halide::Buffer<num_t, 2> A) {
    // We map the original C array A[i][j] to Halide buffer A(j, i):
    //   - i is the row index    -> second dimension (y)
    //   - j is the column index -> first dimension (x)

    Var i("i"), j("j");
    Func A0("A0"), B("B");

    // -------------------------------------------------------------------------
    // Stage 1: build the lower-triangular matrix A0, matching Polybench:
    //
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j <= i; j++)
    //     A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
    //   for (j = i+1; j < n; j++)
    //     A[i][j] = 0;
    //   A[i][i] = 1;
    // }
    //
    // For 0 <= j < n, (-j % n) == -j, so we can drop the % n.
    // Mapping C A[i][j] -> A0(j, i).
    // -------------------------------------------------------------------------
    Expr one  = cast<num_t>(1);
    Expr zero = cast<num_t>(0);
    Expr n_val = cast<num_t>(n);

    A0(j, i) = select(
        j <= i,
        // On or below the diagonal
        select(i == j,
               // Diagonal element
               one,
               // Strictly lower triangular part
               cast<num_t>(-j) / n_val + one),
        // Above the diagonal
        zero);

    // -------------------------------------------------------------------------
    // Stage 2: make the matrix symmetric positive definite:
    //
    // B[r][s] = sum_t A0[r][t] * A0[s][t];
    // then copy B back into A.
    //
    // Mapping (r,s) -> (i,j) in C and to (x,y) = (j,i) in Halide:
    //   B(j, i) = sum_{t=0..n-1} A0(t, i) * A0(t, j)
    // -------------------------------------------------------------------------
    RDom t(0, n, "t");
    B(j, i) = cast<num_t>(0);
    B(j, i) += A0(t, i) * A0(t, j);

    // ---------------------- Scheduling for SPD build -------------------------
    //
    // This is an O(n^3) dense symmetric matrix-multiply. We tile, parallelize,
    // and vectorize over the j (x) dimension, which is contiguous in memory.
    // -------------------------------------------------------------------------
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    Var jo("jo"), io("io"), ji("ji"), ii("ii");

    // Compute B in 32x32 tiles, parallel over tiles, vectorize across columns.
    B.compute_root()
        .tile(j, i, jo, io, ji, ii, 32, 32)
        .fuse(jo, io, jo)        // single tile index for parallelism
        .parallel(jo)
        .vectorize(ji, vec_width);

    // Within each tile, accumulate over t with j (columns) as innermost,
    // so vectorization over ji is effective.
    B.update()
        .reorder(t, ji, ii, jo)
        .vectorize(ji, vec_width);

    // Realize B directly into the provided buffer A. This is equivalent
    // to computing B and then copying B into A as in the original C code.
    B.realize(A);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Problem size
    int n = N;

    // Matrix A: original C is DATA_TYPE A[N][N];
    // We map to Halide Buffer with shape (cols, rows) = (n, n)
    Halide::Buffer<num_t, 2> A(n, n);

    // Initialize A (SPD matrix A = A0 * A0ᵀ)
    init_array(n, A);

    // -------------------------------------------------------------------------
    // Translate kernel_lu into Halide-assisted code.
    //
    // Original C kernel:
    //
    // for (i = 0; i < n; i++) {
    //   for (j = 0; j < i; j++) {
    //     for (k = 0; k < j; k++) {
    //       A[i][j] -= A[i][k] * A[k][j];
    //     }
    //     A[i][j] /= A[j][j];
    //   }
    //   for (j = i; j < n; j++) {
    //     for (k = 0; k < i; k++) {
    //       A[i][j] -= A[i][k] * A[k][j];
    //     }
    //   }
    // }
    //
    // This algorithm has loop-carried dependencies along i and j, so we cannot
    // express the full kernel as a single pure Halide Func. Instead, we:
    //
    //   - Keep the outer loops over i and the L-part j< i on the host (as
    //     required to preserve the sequential dependencies).
    //   - Use Halide Funcs to compute the inner summations over k:
    //       sum_{k<j} A[i][k] * A[k][j]   (L-part)
    //       sum_{k<i} A[i][k] * A[k][j]   (U-part)
    //
    //   - For the U-part (j >= i) we compute the entire updated row i in one
    //     Halide pipeline invocation, improving data locality and amortizing
    //     JIT/launch overhead.
    // -------------------------------------------------------------------------

    // ImageParam representing the current matrix A; note that we never write
    // to A inside Halide; we only read from it in the Funcs below. The
    // in-place updates to A are done on the host after each realization.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    A_param.set(A);

    // Scalar parameters for the current (i, j) indices
    Param<int> i_param("i");
    Param<int> j_param("j");

    // -------------------------------------------------------------------------
    // Func to compute A[i][j] - sum_{k=0}^{j-1} A[i][k] * A[k][j]
    // for the case j < i (the "L" part), before dividing by A[j][j].
    //
    // Mapping:
    //   A[i][j] in C           -> A_param(j_param, i_param) in Halide
    //   A[i][k] (row access)   -> A_param(k_left, i_param)
    //   A[k][j] (column access)-> A_param(j_param, k_left)
    // -------------------------------------------------------------------------
    Func left_elem("left_elem");
    {
        left_elem() = A_param(j_param, i_param);

        // Sum over k = 0 .. j-1:
        // A[i][j] -= A[i][k] * A[k][j]
        RDom k_left(0, j_param, "k_left");
        left_elem() += cast<num_t>(-1) *
                       A_param(k_left, i_param) *
                       A_param(j_param, k_left);
    }

    // -------------------------------------------------------------------------
    // Func to compute the entire updated row i for the "U" part (j >= i):
    //
    //   For fixed i:
    //     for all j:
    //       row_u[j] = A[i][j] - sum_{k=0}^{i-1} A[i][k] * A[k][j]
    //
    // We will realize this as a 1-D buffer over j and then copy back the
    // entries j >= i into A[i][j]. Entries j < i are computed but ignored.
    //
    // Mapping to Halide:
    //   A[i][j]          -> A_param(j, i_param)
    //   A[i][k]          -> A_param(k_row, i_param)
    //   A[k][j]          -> A_param(j, k_row)
    // -------------------------------------------------------------------------
    Var jv("jv");
    Func u_row("u_row");
    {
        u_row(jv) = A_param(jv, i_param);

        RDom k_row(0, i_param, "k_row");
        u_row(jv) += cast<num_t>(-1) *
                     A_param(k_row, i_param) *
                     A_param(jv, k_row);
    }

    // -------------------------- Schedule for u_row ---------------------------
    //
    // For each row i:
    //   - u_row(j) is independent across j.
    //   - We tile and parallelize across j, and vectorize the innermost loop.
    // -------------------------------------------------------------------------
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    Var jo("jo"), ji("ji");
    u_row.compute_root()
         .split(jv, jo, ji, 128)   // tiles of 128 columns
         .parallel(jo)             // parallel over column tiles
         .vectorize(ji, vec_width);

    // JIT-compile the scalar and row pipelines once; we will reuse them.
    left_elem.compile_jit(target);
    u_row.compile_jit(target);

    // Temporary buffer to hold one updated row of U per i
    Halide::Buffer<num_t> u_row_buf(n);

    // -------------------------------------------------------------------------
    // Run kernel and time it
    // -------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        // -----------------------------
        // First inner loop: L-part j=0..i-1
        // -----------------------------
        for (int j = 0; j < i; j++) {
            i_param.set(i);
            j_param.set(j);

            // Compute A[i][j] - sum_{k<j} A[i][k] * A[k][j]
            Halide::Buffer<num_t> tmp = left_elem.realize();
            num_t val = tmp();  // zero-dimensional buffer

            // Divide by A[j][j]
            // Mapping: A[j][j] in C -> A(j, j) in Halide
            val /= A(j, j);

            // Store back into A[i][j] (i.e. A(j, i) in Halide)
            A(j, i) = val;
        }

        // -----------------------------
        // Second inner loop: U-part j=i..n-1
        // Compute full updated row i in one Halide call.
        // -----------------------------
        i_param.set(i);

        // u_row reads the current state of A (including the newly-updated L row i)
        // and computes:
        //   u_row(j) = A[i][j] - sum_{k<i} A[i][k] * A[k][j]
        u_row.realize(u_row_buf);

        // Copy back only the needed entries (j >= i) into A[i][j]
        for (int j = i; j < n; j++) {
            // A[i][j] in C corresponds to A(j, i) in Halide
            A(j, i) = u_row_buf(j);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -------------------------------------------------------------------------
    // Optional: print the resulting matrix, in the same order as the C version
    // -------------------------------------------------------------------------
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // A[i][j] in C corresponds to A(j, i) in Halide
                std::cerr << A(j, i) << '\n';
            }
        }
    }

    // Print timing (in seconds)
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}