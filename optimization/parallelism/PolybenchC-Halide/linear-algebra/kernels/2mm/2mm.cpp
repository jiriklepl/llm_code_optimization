#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

using namespace Halide;

#ifndef NI
#define NI 512
#endif
#ifndef NJ
#define NJ 512
#endif
#ifndef NK
#define NK 512
#endif
#ifndef NL
#define NL 512
#endif

using num_t = float;

static
void init_array(int ni, int nj, int nk, int nl,
                num_t* alpha,
                num_t* beta,
                Halide::Buffer<num_t, 2> A,
                Halide::Buffer<num_t, 2> B,
                Halide::Buffer<num_t, 2> C,
                Halide::Buffer<num_t, 2> D) {
    *alpha = (num_t)1.5;
    *beta = (num_t)1.2;

    Var i("i"), j("j"), k("k"), l("l");
    Func init_A("init_A"), init_B("init_B"), init_C("init_C"), init_D("init_D");

    // A is NI x NK -> Buffer shape (NK, NI) accessed as A(k, i)
    init_A(k, i) = cast<num_t>(((i * k + 1) % ni)) / cast<int>(ni);
    // B is NK x NJ -> Buffer shape (NJ, NK) accessed as B(j, k)
    init_B(j, k) = cast<num_t>((k * (j + 1) % nj)) / cast<int>(nj);
    // C is NJ x NL -> Buffer shape (NL, NJ) accessed as C(l, j)
    init_C(l, j) = cast<num_t>(((j * (l + 3) + 1) % nl)) / cast<int>(nl);
    // D is NI x NL -> Buffer shape (NL, NI) accessed as D(l, i)
    init_D(l, i) = cast<num_t>((i * (l + 2) % nk)) / cast<int>(nk);

    // Simple vectorized/parallel init to speed up setup without affecting the timed kernel
    init_A.vectorize(k, 16).parallel(i);
    init_B.vectorize(j, 16).parallel(k);
    init_C.vectorize(l, 16).parallel(j);
    init_D.vectorize(l, 16).parallel(i);

    init_A.realize(A);
    init_B.realize(B);
    init_C.realize(C);
    init_D.realize(D);
}

int main(int argc, char* argv[]) {
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    num_t alpha;
    num_t beta;

    // Buffers: map C arrays [rows][cols] -> Buffer(cols, rows)
    Buffer<num_t, 2> A(nk, ni);  // A[i][k] -> A(k, i)
    Buffer<num_t, 2> B(nj, nk);  // B[k][j] -> B(j, k)
    Buffer<num_t, 2> C(nl, nj);  // C[j][l] -> C(l, j)
    Buffer<num_t, 2> D(nl, ni);  // D[i][l] -> D(l, i)

    // Initialize arrays
    init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);

    // ImageParams for inputs
    ImageParam A_param(type_of<num_t>(), 2, "A_param");
    ImageParam B_param(type_of<num_t>(), 2, "B_param");
    ImageParam C_param(type_of<num_t>(), 2, "C_param");
    ImageParam D_param(type_of<num_t>(), 2, "D_param");

    A_param.set(A);
    B_param.set(B);
    C_param.set(C);
    D_param.set(D);

    // Vars
    Var i("i"), j("j"), k("k"), l("l");

    // First product: tmp = alpha * A * B, tmp is NI x NJ -> Buffer shape (NJ, NI)
    Func tmp_f("tmp_f");
    RDom r_k(0, A_param.dim(0).extent(), "r_k"); // sum over NK
    tmp_f(j, i) = cast<num_t>(0.0f);
    tmp_f(j, i) += Expr(alpha) * A_param(r_k, i) * B_param(j, r_k);
    tmp_f.reorder(j, i);
    tmp_f.update().reorder(j, r_k, i);

    // Second product and accumulation: D = beta*D + tmp * C
    // D is NI x NL -> Buffer shape (NL, NI) with coords (l, i)
    Func D_out("D_out");
    RDom r_j(0, C_param.dim(1).extent(), "r_j"); // sum over NJ
    D_out(l, i) = Expr(beta) * D_param(l, i);
    D_out(l, i) += tmp_f(r_j, i) * C_param(l, r_j);
    D_out.reorder(l, i);
    D_out.update().reorder(l, r_j, i);

    // Optimized schedule focusing on data locality and memory bandwidth
    const int vec = 8;    // vector width in l or j
    const int i_blk = 32; // tile size for i (rows)

    Var io("io"), ii("ii"), lo("lo"), li("li");

    // Compute final output at root, tile across i and vectorize across l (contiguous)
    D_out.compute_root()
         .split(i, io, ii, i_blk)
         .split(l, lo, li, vec)
         .reorder(li, ii, lo, io)
         .vectorize(li)
         .parallel(io);

    // Match the same tiling strategy for the update definition
    D_out.update()
         .split(i, io, ii, i_blk)
         .split(l, lo, li, vec)
         // Keep l innermost (vectorized), then reduce over r_j for good C(l, r_j) access
         .reorder(li, r_j, ii, lo, io)
         .vectorize(li)
         .parallel(io);

    // Compute tmp per i-tile to improve reuse and locality; vectorize across j (contiguous in B and tmp)
    tmp_f.compute_at(D_out, ii);
    tmp_f.vectorize(j, vec);
    tmp_f.update().vectorize(j, vec);

    // JIT compile for the host
    Target target = get_host_target();
    D_out.compile_jit(target);

    auto start = std::chrono::high_resolution_clock::now();

    // Realize into D (updates in place logically)
    D_out.realize(D);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // Print results to stderr (one value per line)
    using namespace std::string_literals;
    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(2);
        for (int ii2 = 0; ii2 < ni; ii2++) {
            for (int jj2 = 0; jj2 < nl; jj2++) {
                std::cerr << D(jj2, ii2) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}
