#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

#include "defines.hpp"
#include "heat-3d.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

static
void init_array(int n,
                Halide::Buffer<num_t, 3> A,
                Halide::Buffer<num_t, 3> B) {
    Var k("k"), j("j"), i("i");
    Func init("init");

    // A[i][j][k] = (i + j + (n - k)) * 10 / n;
    // Buffer layout: (k, j, i)
    Param<int> pn("pn");
    init(k, j, i) = cast<num_t>((i + j + (pn - k)) * 10) / cast<num_t>(pn);

    pn.set(n);
    init.realize(A);
    init.realize(B);
}

int main(int argc, char *argv[]) {
    // problem size
    int n = N;
    int tsteps = TSTEPS;

    // Buffers (layout: (k, j, i))
    Halide::Buffer<num_t, 3> A(n, n, n);
    Halide::Buffer<num_t, 3> B(n, n, n);

    // initialize data
    init_array(n, A, B);

    // Define one stencil step: out = f(in)
    ImageParam In(type_of<num_t>(), 3, "In");

    Var k("k"), j("j"), i("i");
    Func step("step");

    // Constants as Exprs of correct type
    Expr c0125 = cast<num_t>(Expr(0.125));
    Expr c2    = cast<num_t>(Expr(2.0));

    Expr center = In(k, j, i);
    Expr right  = In(k + 1, j, i);
    Expr left   = In(k - 1, j, i);
    Expr down   = In(k, j + 1, i);
    Expr up     = In(k, j - 1, i);
    Expr front  = In(k, j, i + 1);
    Expr back   = In(k, j, i - 1);

    Expr interior =
        c0125 * (right - c2 * center + left) +
        c0125 * (down  - c2 * center + up)   +
        c0125 * (front - c2 * center + back) +
        center;

    // Only update interior points; keep boundaries unchanged
    Expr cond =
        (k > 0) && (k < In.dim(0).extent() - 1) &&
        (j > 0) && (j < In.dim(1).extent() - 1) &&
        (i > 0) && (i < In.dim(2).extent() - 1);

    step(k, j, i) = select(cond, interior, center);

    // Scheduling: improve locality and vectorization
    // - The buffer layout is (k, j, i), so k is the innermost, contiguous dimension.
    // - Tile across i and j; fuse tiles for coarse-grain parallelism.
    // - Vectorize across k to use SIMD on the fast-varying dim.
    Var io("io"), jo("jo"), ii("ii"), ji("ji"), tile("tile");

    const int vec = (sizeof(num_t) == 4) ? 8   // float: try 8-wide if available
                                         : 4;  // double/int64: 4-wide

    // Constrain the bounds to help the compiler reason about loops.
    step
        .bound(k, 0, n)
        .bound(j, 0, n)
        .bound(i, 0, n)
        .compute_root()
        .tile(i, j, io, jo, ii, ji, 32, 8)
        .fuse(io, jo, tile)
        .reorder(k, ii, ji, tile)
        .vectorize(k, vec)
        .parallel(tile);

    // The input is contiguous in k; assert stride-1 along k to enable efficient vector loads.
    In.dim(0).set_stride(1);
    In.dim(0).set_bounds(0, n);
    In.dim(1).set_bounds(0, n);
    In.dim(2).set_bounds(0, n);

    // JIT-compile once for the current host target.
    Target t = get_jit_target_from_environment();
    step.compile_jit(t);

    // Time the kernel: perform tsteps of (A->B) then (B->A)
    auto start = std::chrono::high_resolution_clock::now();

    for (int tstep = 1; tstep <= tsteps; tstep++) {
        In.set(A);
        step.realize(B);

        In.set(B);
        step.realize(A);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // print results
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < n; ii++) {
            for (int jj = 0; jj < n; jj++) {
                for (int kk = 0; kk < n; kk++) {
                    std::cerr << A(kk, jj, ii) << '\n';
                }
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}
