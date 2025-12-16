#include <chrono>
#include <iomanip>
#include <iostream>

#include "Halide.h"

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "gemm.hpp"

using num_t = DATA_TYPE;

using namespace Halide;


static
void init_array(int ni, int nj, int nk,
		num_t *alpha,
		num_t *beta,
		Halide::Buffer<num_t, 2> C,
		Halide::Buffer<num_t, 2> A,
		Halide::Buffer<num_t, 2> B) {
	*alpha = (num_t)1.5;
	*beta = (num_t)1.2;

	Var i("i"), j("j"), k("k");
	Func init_A("init_A"), init_B("init_B"), init_C("init_C");

	init_C(j, i) = cast<num_t>((i * j + 1) % ni) / cast<int>(ni);
	init_A(k, i) = cast<num_t>(i * (k + 1) % nk) / cast<int>(nk);
	init_B(j, k) = cast<num_t>(k * (j + 2) % nj) / cast<int>(nj);

	init_C.realize(C);
	init_A.realize(A);
	init_B.realize(B);
}

int main(int argc, char *argv[]) {
	// problem size
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	// input data
	num_t alpha;
	num_t beta;

	Buffer<num_t, 2> C(nj, ni);
	Buffer<num_t, 2> B(nj, nk);
	Buffer<num_t, 2> A(nk, ni);

	ImageParam C_param(type_of<num_t>(), 2, "C_param");
	ImageParam A_param(type_of<num_t>(), 2, "A_param");
	ImageParam B_param(type_of<num_t>(), 2, "B_param");

	// initialize data
	init_array(ni, nj, nk, &alpha, &beta, C, A, B);

	Var i("i"), j("j");
	RDom k(0, A.dim(0).extent(), "k");
	Func gemm("gemm");

	gemm(j, i) = Expr(beta) * C_param(j, i);
	gemm(j, i) += Expr(alpha) * A_param(k, i) * B_param(j, k);

	gemm.reorder(j, i);
	gemm.update().reorder(j, k, i);

	C_param.set(C);
	A_param.set(A);
	B_param.set(B);

	// compile the kernel to machine code
	gemm.compile_jit();

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	gemm.realize(C);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	using namespace std::string_literals;
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		for (int i = 0; i < ni; i++) {
			for (int j = 0; j < nj; j++) {
				std::cerr << C(j, i) << '\n';
			}
		}
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}
