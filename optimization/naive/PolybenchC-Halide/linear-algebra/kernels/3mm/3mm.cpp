#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common PolyBench-style definitions
#include "defines.hpp"

// include benchmark-specific definitions (NI, NJ, NK, NL, NM, DATA_TYPE, etc.)
#include "3mm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/**
 * Initialize arrays A, B, C, D with the same values as the original C kernel.
 *
 * C layout (row-major):
 *   A[ni][nk]
 *   B[nk][nj]
 *   C[nj][nm]
 *   D[nm][nl]
 *
 * Halide layout (column-major / x-fastest):
 *   A(k, i) => A[i][k]   -> Buffer A(nk, ni)
 *   B(j, k) => B[k][j]   -> Buffer B(nj, nk)
 *   C(m, j) => C[j][m]   -> Buffer C(nm, nj)
 *   D(l, m) => D[m][l]   -> Buffer D(nl, nm)
 */
static void init_array(int ni, int nj, int nk, int nl, int nm,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B,
                       Halide::Buffer<num_t, 2> C,
                       Halide::Buffer<num_t, 2> D) {
    Var i("i"), j("j"), k("k"), l("l"), m("m");

    Func init_A("init_A"), init_B("init_B"), init_C("init_C"), init_D("init_D");

    // A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
    // A has shape [ni][nk] in C => A(k, i) in Halide
    init_A(k, i) = cast<num_t>(((i * k + 1) % ni)) / cast<int>(5 * ni);

    // B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
    // B has shape [nk][nj] in C => B(j, k) in Halide
    init_B(j, k) = cast<num_t>(((k * (j + 1) + 2) % nj)) / cast<int>(5 * nj);

    // C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
    // C has shape [nj][nm] in C => C(m, j) in Halide
    init_C(m, j) = cast<num_t>(((j * (m + 3)) % nl)) / cast<int>(5 * nl);

    // D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
    // D has shape [nm][nl] in C => D(l, m) in Halide
    init_D(l, m) = cast<num_t>(((m * (l + 2) + 2) % nk)) / cast<int>(5 * nk);

    init_A.realize(A);
    init_B.realize(B);
    init_C.realize(C);
    init_D.realize(D);
}

int main(int argc, char *argv[]) {
    // Problem sizes from 3mm.hpp
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int nm = NM;

    // Allocate Buffers corresponding to the original C arrays.
    //
    // C layout: A[ni][nk] -> Halide: A(nk, ni)  (x = k, y = i)
    //           B[nk][nj] -> B(nj, nk)          (x = j, y = k)
    //           C[nj][nm] -> C(nm, nj)          (x = m, y = j)
    //           D[nm][nl] -> D(nl, nm)          (x = l, y = m)
    //           G[ni][nl] -> G(nl, ni)          (x = l, y = i)
    Halide::Buffer<num_t, 2> A(nk, ni);
    Halide::Buffer<num_t, 2> B(nj, nk);
    Halide::Buffer<num_t, 2> C(nm, nj);
    Halide::Buffer<num_t, 2> D(nl, nm);
    Halide::Buffer<num_t, 2> G(nl, ni);  // final result

    // Initialize input matrices A, B, C, D.
    init_array(ni, nj, nk, nl, nm, A, B, C, D);

    // Create ImageParams for the input matrices so that the Halide
    // pipeline can be compiled independently of specific Buffers.
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam D_param(type_of<num_t>(), 2, "D_param");

    A_param.set(A);
    B_param.set(B);
    C_param.set(C);
    D_param.set(D);

    // Vars and reduction domains.
    Var i("i"), j("j"), l("l");
    // One-dimensional reduction variables: we will use rAB.x, rCD.x, rEF.x below.
    RDom rAB(0, nk, "rAB");  // reduction over k for E = A * B
    RDom rCD(0, nm, "rCD");  // reduction over k for F = C * D
    RDom rEF(0, nj, "rEF");  // reduction over k for G = E * F

    // Define Funcs for the three matrix multiplications.
    //
    // 1) E := A * B
    // C: E[ni][nj] = sum_{k=0..nk-1} A[i][k] * B[k][j]
    // Halide: E(j, i) = sum_k A_param(k, i) * B_param(j, k)
    Func E("E");
    E(j, i) = cast<num_t>(0);
    E(j, i) += A_param(rAB.x, i) * B_param(j, rAB.x);

    // 2) F := C * D
    // C: F[nj][nl] = sum_{k=0..nm-1} C[i][k] * D[k][j]
    // Halide: F(l, j) = sum_k C_param(k, j) * D_param(l, k)
    Func F("F");
    F(l, j) = cast<num_t>(0);
    F(l, j) += C_param(rCD.x, j) * D_param(l, rCD.x);

    // 3) G := E * F
    // C: G[ni][nl] = sum_{k=0..nj-1} E[i][k] * F[k][j]
    // Halide: G_func(l, i) = sum_k E(k, i) * F(l, k)
    Func G_func("G");
    G_func(l, i) = cast<num_t>(0);
    G_func(l, i) += E(rEF.x, i) * F(l, rEF.x);

    // --------------------------------------------------------------------
    // Optimized schedule
    //
    // Goal:
    //   - Keep the inner (x) dimension contiguous to enable vectorization.
    //   - Parallelize over outer (row) dimensions to use all CPU cores.
    //   - Preserve original math and memory layouts.
    //
    // All three matrix multiplies have the same general pattern:
    //   for each output row (i / j), loop over vectorized columns (x),
    //   and for each reduction k, accumulate into a vector of outputs.
    //
    // We choose a fixed vector width of 8, which maps well to AVX on
    // most x86 CPUs for float; on machines with smaller SIMD width,
    // Halide will lower this to multiple narrower vectors or scalars.
    // --------------------------------------------------------------------
    const int vec = 8;

    // Schedule for E: E(j, i)  -- j is contiguous in memory (x dimension), i is rows (y).
    {
        Var jo("jo"), ji("ji");

        // Compute E at the root: full matrix stored.
        E.compute_root();

        // Pure definition: initialize E(j,i) = 0
        //   Split j so that ji is a small, vectorized inner loop.
        //   Loop nest:
        //     for i (parallel)
        //       for jo
        //         for ji (vectorized)
        E.split(j, jo, ji, vec)
         .reorder(ji, i, jo)
         .vectorize(ji)   // operate on vec contiguous columns of j at once
         .parallel(i);    // distribute rows of E across threads

        // Update definition: E(j,i) += A(k,i) * B(j,k)
        //   We keep the reduction over k (rAB.x) inside the row loop,
        //   but outside the vectorized j loop, so that for each k we:
        //     - load A_param(k, i) once (scalar, reused across ji),
        //     - load a vector B_param(j_base + ji, k) contiguously,
        //     - update a vector of E.
        E.update()
         .split(j, jo, ji, vec)
         .reorder(ji, rAB.x, i, jo)
         .vectorize(ji)
         .parallel(i);
    }

    // Schedule for F: F(l, j)  -- l is contiguous in memory (x dimension), j is rows (y).
    {
        Var lo("lo"), li("li");

        F.compute_root();

        // Pure definition: F(l,j) = 0
        F.split(l, lo, li, vec)
         .reorder(li, j, lo)
         .vectorize(li)   // vector over contiguous l
         .parallel(j);    // parallel over rows j

        // Update definition: F(l,j) += C(k,j) * D(l,k)
        F.update()
         .split(l, lo, li, vec)
         .reorder(li, rCD.x, j, lo)
         .vectorize(li)
         .parallel(j);
    }

    // Schedule for G: G_func(l, i)  -- l is contiguous in memory (x dimension), i is rows (y).
    {
        Var lo("lo"), li("li");

        G_func.compute_root();

        // Pure definition: G(l,i) = 0
        G_func.split(l, lo, li, vec)
              .reorder(li, i, lo)
              .vectorize(li)   // vector over contiguous l
              .parallel(i);    // parallel over rows i

        // Update definition: G(l,i) += E(k,i) * F(l,k)
        //   For each i and k, we:
        //     - load scalar E(k,i),
        //     - load a vector F(l_base + li, k) contiguously,
        //     - update a vector of G(l,i).
        G_func.update()
              .split(l, lo, li, vec)
              .reorder(li, rEF.x, i, lo)
              .vectorize(li)
              .parallel(i);
    }

    // Compile the kernel to machine code (JIT).
    Target target = get_jit_target_from_environment();
    G_func.compile_jit(target);

    // Measure execution time (kernel only).
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel: realize G_func into Buffer G (shape nl x ni).
    G_func.realize(G);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Optionally print the result matrix G to stderr.
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii = 0; ii < ni; ii++) {
            for (int jj = 0; jj < nl; jj++) {
                // G[ii][jj] in C => G(jj, ii) in Halide
                std::cerr << G(jj, ii) << '\n';
            }
        }
    }

    // Print timing in seconds to stdout with 6 decimal places,
    // matching the style of the gemm.cpp example.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}