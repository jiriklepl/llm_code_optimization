#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common Polybench-like definitions (DATA_TYPE, N, TSTEPS, etc)
#include "defines.hpp"

// include benchmark-specific definitions (if any)
#include "jacobi-2d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Halide-based initialization of A and B.
 *
 * Original C code:
 *
 *   for (i = 0; i < n; i++)
 *     for (j = 0; j < n; j++) {
 *       A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
 *       B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
 *     }
 *
 * We map A[i][j] -> A(j, i) and B[i][j] -> B(j, i) in Halide.
 *
 * This version adds a schedule that:
 *   - computes both init Funcs at root
 *   - parallelizes over rows (i)
 *   - vectorizes across columns (j) using the natural SIMD width
 */
static
void init_array(int n,
                Halide::Buffer<num_t, 2> A,
                Halide::Buffer<num_t, 2> B) {
    Var i("i"), j("j");
    Func init_A("init_A"), init_B("init_B");

    // n as a Halide Expr, with the correct element type.
    Expr n_expr = cast<num_t>(Expr(n));

    // A(i,j) in C -> A(j,i) in Halide
    init_A(j, i) = cast<num_t>(i * (j + 2) + 2) / n_expr;
    // B(i,j) in C -> B(j,i) in Halide
    init_B(j, i) = cast<num_t>(i * (j + 3) + 3) / n_expr;

    // -------------------------------
    // Schedule for initialization
    // -------------------------------
    // Use the host JIT target to pick a reasonable vector width.
    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // Compute both initializations at root, parallelizing over rows (i)
    // and vectorizing across columns (j). This matches row-major storage
    // (j is the fast-varying x dimension) and gives good cache behavior.
    init_A
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    init_B
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // JIT-compile once with the chosen target; realize() will reuse this.
    init_A.compile_jit(target);
    init_B.compile_jit(target);

    // Actually fill A and B.
    init_A.realize(A);
    init_B.realize(B);
}

int main(int argc, char *argv[]) {
    // Problem size
    int n       = N;
    int tsteps  = TSTEPS;

    // Buffers corresponding to A[N][N] and B[N][N].
    // C layout is A[i][j]; we map that to A_buf(j, i).
    Halide::Buffer<num_t, 2> A(n, n);  // x = j, y = i
    Halide::Buffer<num_t, 2> B(n, n);  // x = j, y = i

    // Initialize A and B.
    init_array(n, A, B);

    // -------------------------------
    // Define the Jacobi 2D stencil in Halide
    // -------------------------------

    // JIT target and natural vector width, used for scheduling.
    Target target = get_jit_target_from_environment();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    // ImageParam representing the "current" grid (either A or B).
    ImageParam cur_param(type_of<num_t>(), 2, "cur_param");

    // Vars: x corresponds to column index j, y to row index i.
    Var x("x"), y("y");

    // Factor 0.2 in the correct type.
    Expr w = cast<num_t>(Expr(0.2));

    // Jacobi update:
    //
    // B[i][j] = 0.2 * (A[i][j] +
    //                  A[i][j-1] + A[i][j+1] +
    //                  A[i+1][j] + A[i-1][j]);
    //
    // In Halide coordinates (x=j, y=i):
    // next(x,y) = 0.2 * (cur(x,y) +
    //                    cur(x-1,y) + cur(x+1,y) +
    //                    cur(x,y+1) + cur(x,y-1));
    Func jacobi("jacobi");
    jacobi(x, y) = w * (cur_param(x, y) +
                        cur_param(x - 1, y) +
                        cur_param(x + 1, y) +
                        cur_param(x, y + 1) +
                        cur_param(x, y - 1));

    // -------------------------------
    // Schedule for the Jacobi step
    // -------------------------------
    //
    // The domain we realize over is the interior: x,y = 1 .. n-2.
    // We tile that region to improve locality, vectorize across x,
    // and parallelize over tiles in y.
    //
    // The schedule is generic in n: Halide generates loops whose
    // bounds depend on the realized Buffer's size, so this works
    // for any N at runtime.
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    // Choose tile sizes; make x tiles a multiple of the SIMD width.
    const int tile_x = vec_width * 32;  // wide enough to amortize overhead
    const int tile_y = 32;              // reasonable strip-height

    jacobi
        .tile(x, y, xo, yo, xi, yi, tile_x, tile_y)
        .vectorize(xi, vec_width)  // SIMD across contiguous x
        .parallel(yo);             // parallelize across tiles in y

    // A small unroll in y inside the tile can help ILP without
    // exploding code size; it is safe even when tile_y isn't
    // a multiple of the factor.
    jacobi.unroll(yi, 2);

    // We will only apply the stencil to the interior points:
    // i,j = 1 .. n-2. The boundaries remain unchanged across time steps,
    // matching the original C code.
    //
    // Create interior views of A and B that alias the underlying storage.
    // Dimension 0 is x (j), dimension 1 is y (i).
    Halide::Buffer<num_t, 2> A_interior =
        A.cropped(0, 1, n - 2).cropped(1, 1, n - 2);
    Halide::Buffer<num_t, 2> B_interior =
        B.cropped(0, 1, n - 2).cropped(1, 1, n - 2);

    // Bind some initial buffer (say A) before compiling.
    cur_param.set(A);
    // Compile the Jacobi step once with the chosen target; we'll
    // reuse it for all time steps by rebinding cur_param.
    jacobi.compile_jit(target);

    // -------------------------------
    // Time-stepping loop, driven from the host
    // -------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // Original C kernel:
    //
    // for (t = 0; t < tsteps; t++) {
    //   // B from A
    //   for (i = 1; i < n-1; i++)
    //     for (j = 1; j < n-1; j++)
    //       B[i][j] = stencil(A,...);
    //
    //   // A from B
    //   for (i = 1; i < n-1; i++)
    //     for (j = 1; j < n-1; j++)
    //       A[i][j] = stencil(B,...);
    // }
    //
    // We implement the same ping-pong using a single Halide Func "jacobi":
    // - Each call to jacobi.realize applies the 5-point stencil to
    //   interior points only.
    // - Array boundaries are never written inside this loop and thus
    //   retain their initial values, exactly like the C reference.
    for (int t = 0; t < tsteps; t++) {
        // First phase: B <- stencil(A)
        cur_param.set(A);
        jacobi.realize(B_interior);

        // Second phase: A <- stencil(B)
        cur_param.set(B);
        jacobi.realize(A_interior);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // -------------------------------
    // Optional: print the final A to stderr
    // (similar intent to polybench_prevent_dce(print_array))
    // -------------------------------
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Recall A(i,j) in C is A(j,i) in Halide.
                std::cerr << A(j, i) << '\n';
            }
        }
    }

    // Print elapsed time in seconds to stdout, with microsecond precision.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}