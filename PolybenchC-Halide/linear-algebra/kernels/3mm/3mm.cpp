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
    // C layout: A[ni][nk] -> Halide: A(nk, ni)  (x=j/col, y=i/row)
    //           B[nk][nj] -> B(nj, nk)
    //           C[nj][nm] -> C(nm, nj)
    //           D[nm][nl] -> D(nl, nm)
    //           G[ni][nl] -> G(nl, ni)
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
    RDom rAB(0, nk, "rAB");  // reduction over k for E = A * B
    RDom rCD(0, nm, "rCD");  // reduction over k for F = C * D
    RDom rEF(0, nj, "rEF");  // reduction over k for G = E * F

    // Define Funcs for the three matrix multiplications.
    //
    // 1) E := A * B
    // C: E[ni][nj] = sum_{k=0..nk-1} A[i][k] * B[k][j]
    // Halide: E(j, i)
    Func E("E");
    E(j, i) = cast<num_t>(0);
    E(j, i) += A_param(rAB, i) * B_param(j, rAB);

    // 2) F := C * D
    // C: F[nj][nl] = sum_{k=0..nm-1} C[i][k] * D[k][j]
    // Halide: F(l, j)
    Func F("F");
    F(l, j) = cast<num_t>(0);
    F(l, j) += C_param(rCD, j) * D_param(l, rCD);

    // 3) G := E * F
    // C: G[ni][nl] = sum_{k=0..nj-1} E[i][k] * F[k][j]
    // Halide: G_func(l, i)
    Func G_func("G");
    G_func(l, i) = cast<num_t>(0);
    G_func(l, i) += E(rEF, i) * F(l, rEF);

    // Provide a simple schedule that mirrors the original loop structure.

    // E(i,j,k): outer over i (rows), then j (cols), inner reduction over k.
    E.compute_root();
    E.reorder(j, i);
    E.update().reorder(j, rAB, i);

    // F(i,j,k): outer over i (rows -> j here), then j (cols -> l), inner reduction over k.
    F.compute_root();
    F.reorder(l, j);
    F.update().reorder(l, rCD, j);

    // G(i,j,k): outer over i (rows -> i), then j (cols -> l), inner reduction over k.
    G_func.compute_root();
    G_func.reorder(l, i);
    G_func.update().reorder(l, rEF, i);

    // Compile the kernel to machine code.
    G_func.compile_jit();

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