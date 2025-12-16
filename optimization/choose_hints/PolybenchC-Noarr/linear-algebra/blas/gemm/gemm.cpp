#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

// base 1D layouts
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

// matrix layouts
struct tuning {
	// C: i x j
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: k x j
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);
} tuning;

// Array initialization
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// cache lengths outside traversers
	const auto ni = C | get_length<'i'>();
	const auto nk = A | get_length<'k'>();
	const auto nj = B | get_length<'j'>();

	// C[i][j] = (i*j+1) % ni / ni
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 1) % ni) / ni;
	});

	// A[i][k] = i*(k+1) % nk / nk
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)(i * (k + 1) % nk) / nk;
	});

	// B[k][j] = k*(j+2) % nj / nj
	traverser(B).for_each([=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 2) % nj) / nj;
	});
}

// Main computational kernel (GEMM)
// Form C := alpha*A*B + beta*C,
// A is NIxNK, B is NKxNJ, C is NIxNJ
//
// Optimized for locality:
// 1) Scale C by beta in a separate pass.
// 2) Use i–k–j loop order:
//      for i
//        for k
//          a_ik = A[i][k]
//          for j
//            C[i][j] += alpha * a_ik * B[k][j];
//    This keeps the innermost j-loop contiguous for both B and C, and
//    reloads A[i][k] only once per (i,k) pair.
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

#pragma scop
	// 1) Scale C: C[i][j] *= beta
	traverser(C).for_each([=](auto state) {
		C[state] *= beta;
	});

	// 2) GEMM update: C[i][j] += alpha * A[i][k] * B[k][j]
	//
	// We traverse all (i,k) pairs explicitly via for_dims<'i','k'>().
	// For each such pair we:
	//   - load A[i][k] once
	//   - run an inner j-loop (for_each) that walks B[k][j] and C[i][j]
	//     in their contiguous j dimension.
	traverser(C, A, B).template for_dims<'i', 'k'>([=](auto ik_trav) {
		// Current (i,k) indices
		auto state_ik = ik_trav.state();

		// Preload A[i][k] once for this (i,k)
		const num_t a_ik = A[state_ik];
		const num_t alpha_a_ik = alpha * a_ik;

		// Iterate over all j for this fixed (i,k)
		ik_trav.for_each([=](auto state) {
			// state contains (i,j,k); j varies here
			// C[i][j] += alpha * A[i][k] * B[k][j]
			C[state] += alpha_a_ik * B[state];
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;

	// scalars
	num_t alpha;
	num_t beta;

	// lengths proto-structure (applied to all matrices)
	auto set_lengths =
		noarr::set_length<'i'>(ni)
		^ noarr::set_length<'j'>(nj)
		^ noarr::set_length<'k'>(nk);

	// bags for C, A, B with appropriate layouts
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths); // i x j
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths); // i x k
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths); // k x j

	// initialize arrays
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// output to prevent dead-code elimination (similar role as print_array/polybench_prevent_dce)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}