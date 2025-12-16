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
 * Halide layout (column-major / x-fastest) used here:
 *
 *   We store the row-major matrices in Buffers so that the *column* index
 *   (the second C index) is the Halide x-dimension (contiguous), and the
 *   row index is the Halide y-dimension:
 *
 *     A_buf(k, i)  ≡ A[i][k]   -> Buffer A(nk, ni)
 *     B_buf(k, j)  ≡ B[k][j]   -> Buffer B(nk, nj)   (B stored transposed vs. 3mm.cpp)
 *     C_buf(m, j)  ≡ C[j][m]   -> Buffer C(nm, nj)
 *     D_buf(m, l)  ≡ D[m][l]   -> Buffer D(nm, nl)   (D stored transposed vs. 3mm.cpp)
 *
 * This layout makes the reduction dimensions (k for A*B, m for C*D)
 * contiguous in memory for both multiplicands, which improves cache
 * and vectorization behavior without changing the mathematical result.
 */
static void init_array(int ni, int nj, int nk, int nl, int nm,
                       Halide::Buffer<num_t, 2> A,
                       Halide::Buffer<num_t, 2> B,
                       Halide::Buffer<num_t, 2> C,
                       Halide::Buffer<num_t, 2> D) {
    Var i("i"), j("j"), k("k"), l("l"), m("m");

    Func init_A("init_A"), init_B("init_B"), init_C("init_C"), init_D("init_D");

    // A[i][k] = (DATA_TYPE) ((i*k+1) % ni) / (5*ni);
    // A has shape [ni][nk] in C => A(k, i) in Halide.
    init_A(k, i) = cast<num_t>(((i * k + 1) % ni)) / cast<int>(5 * ni);

    // B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
    // B has shape [nk][nj] in C => we store B_buf(k, j) == B[i=k][j].
    init_B(k, j) = cast<num_t>(((k * (j + 1) + 2) % nj)) / cast<int>(5 * nj);

    // C[j][m] = (DATA_TYPE) (j*(m+3) % nl) / (5*nl);
    // C has shape [nj][nm] in C => C(m, j) in Halide.
    init_C(m, j) = cast<num_t>(((j * (m + 3)) % nl)) / cast<int>(5 * nl);

    // D[m][l] = (DATA_TYPE) ((m*(l+2)+2) % nk) / (5*nk);
    // D has shape [nm][nl] in C => we store D(m, l) == D[m][l].
    init_D(m, l) = cast<num_t>(((m * (l + 2) + 2) % nk)) / cast<int>(5 * nk);

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
    // C layout: A[ni][nk] -> Halide: A(k, i)  -> Buffer A(nk, ni)
    //           B[nk][nj] -> Halide: B(k, j)  -> Buffer B(nk, nj)  (transposed vs. 3mm.cpp)
    //           C[nj][nm] -> Halide: C(m, j)  -> Buffer C(nm, nj)
    //           D[nm][nl] -> Halide: D(m, l)  -> Buffer D(nm, nl)  (transposed vs. 3mm.cpp)
    //           G[ni][nl] -> Halide: G(l, i)  -> Buffer G(nl, ni)
    Halide::Buffer<num_t, 2> A(nk, ni);
    Halide::Buffer<num_t, 2> B(nk, nj);
    Halide::Buffer<num_t, 2> C(nm, nj);
    Halide::Buffer<num_t, 2> D(nm, nl);
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
    // rAB: k-reduction for E = A * B
    RDom rAB(0, nk, "rAB");
    // rCD: m-reduction for F = C * D
    RDom rCD(0, nm, "rCD");
    // rEF: j-reduction for G = E * F
    RDom rEF(0, nj, "rEF");

    // Define Funcs for the three matrix multiplications.
    //
    // 1) E := A * B
    // C: E[ni][nj] = sum_{k=0..nk-1} A[i][k] * B[k][j]
    // Halide: E(j, i) holds E[i][j]
    Func E("E");
    E(j, i) = cast<num_t>(0);
    // A_param(rAB, i)   == A[i][k]
    // B_param(rAB, j)   == B[k][j] (we stored B transposed for better locality)
    E(j, i) += A_param(rAB, i) * B_param(rAB, j);

    // 2) F := C * D
    // C: F[nj][nl] = sum_{m=0..nm-1} C[j][m] * D[m][l]
    // We store F transposed as F_T[l][j], so Halide: F(l, j) holds F[j][l]
    Func F("F");
    F(l, j) = cast<num_t>(0);
    // C_param(rCD, j)   == C[j][m]
    // D_param(rCD, l)   == D[m][l] (we stored D transposed for better locality)
    F(l, j) += C_param(rCD, j) * D_param(rCD, l);

    // 3) G := E * F
    // C: G[ni][nl] = sum_{j=0..nj-1} E[i][j] * F[j][l]
    // Recall: E(j, i) == E[i][j], F(l, j) == F[j][l]
    // Halide: G_func(l, i) holds G[i][l]
    Func G_func("G");
    G_func(l, i) = cast<num_t>(0);
    G_func(l, i) += E(rEF, i) * F(l, rEF);

    // ----------------------------------------------------------------------
    // Scheduling for performance
    // ----------------------------------------------------------------------
    // Strategy:
    //   - Materialize E and F at root (as in the original PolyBench kernel).
    //   - Parallelize across rows (i) / j-rows where safe.
    //   - Vectorize across contiguous x-dimension:
    //         E: x = j, F: x = l, G: x = l.
    //   - Optionally tile the outer i/j dimensions to improve cache locality.
    //
    // Data layout choices above make the reduction variables rAB (k) and
    // rCD (m) contiguous in memory for both multiplicands, which improves
    // cache reuse even without explicit blocking in the reduction domain.

    // Query a reasonable vector width for the current target CPU and data type.
    Target target = get_jit_target_from_environment();
    int vec_width = target.natural_vector_size(type_of<num_t>());

    // Tile sizes for the outer spatial dimensions (rows/cols). These are
    // modest and chosen to be multiples of typical vector widths.
    const int tile_i = 32;
    const int tile_j = 32;

    // ----------------------
    // Schedule for E = A * B
    // ----------------------
    // E(j, i): j is x (contiguous), i is y.

    Var i_outer_E("i_outer_E"), i_inner_E("i_inner_E");

    // Tile the i-dimension to get coarse-grain parallelism over row blocks.
    E.split(i, i_outer_E, i_inner_E, tile_i);
    // Make the per-tile loop order: i_outer_E (rows), then i_inner_E (rows
    // within tile), and j innermost for contiguous vectorized writes.
    E.reorder(j, i_inner_E, i_outer_E);
    E.parallel(i_outer_E);
    E.vectorize(j, vec_width);

    // For the reduction update, we use the same split on i and reorder so
    // that:
    //   outer:  i_outer_E
    //   inner:  i_inner_E
    //   innermost: j (vectorized), with rAB between j and i_inner_E
    E.update()
        .reorder(j, rAB, i_inner_E, i_outer_E)
        .parallel(i_outer_E)
        .vectorize(j, vec_width);

    // ----------------------
    // Schedule for F = C * D
    // ----------------------
    // F(l, j): l is x (contiguous), j is y.

    Var j_outer_F("j_outer_F"), j_inner_F("j_inner_F");

    // Tile the j dimension to parallelize over row blocks of F.
    F.split(j, j_outer_F, j_inner_F, tile_j);
    // Order: j_outer_F (rows of C/F), then j_inner_F, with l innermost.
    F.reorder(l, j_inner_F, j_outer_F);
    F.parallel(j_outer_F);
    F.vectorize(l, vec_width);

    // Reduction update: keep rCD between l and j_inner_F, parallelize rows,
    // vectorize along l (contiguous).
    F.update()
        .reorder(l, rCD, j_inner_F, j_outer_F)
        .parallel(j_outer_F)
        .vectorize(l, vec_width);

    // ----------------------
    // Schedule for G = E * F
    // ----------------------
    // G_func(l, i): l is x (contiguous), i is y.

    Var i_outer_G("i_outer_G"), i_inner_G("i_inner_G");

    G_func.split(i, i_outer_G, i_inner_G, tile_i);
    // Order: i_outer_G (rows), then i_inner_G, with l innermost
    // for contiguous vectorized writes.
    G_func.reorder(l, i_inner_G, i_outer_G);
    G_func.parallel(i_outer_G);
    G_func.vectorize(l, vec_width);

    // Reduction update: for each (i_outer_G, i_inner_G, l), accumulate over j.
    G_func.update()
        .reorder(l, rEF, i_inner_G, i_outer_G)
        .parallel(i_outer_G)
        .vectorize(l, vec_width);

    // Compile the kernel to machine code.
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