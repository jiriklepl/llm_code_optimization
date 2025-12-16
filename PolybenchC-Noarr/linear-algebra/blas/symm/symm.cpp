#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "symm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// C: i x j
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k  (M x M)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: i x j  (M x N)
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	auto m = C | get_length<'i'>();
	auto n = C | get_length<'j'>();

	// Initialize C and B: for all i, j
	traverser(C, B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		C[state] = (num_t)((i + j) % 100) / m;
		B[state] = (num_t)((n + i - j) % 100) / m;
	});

	// Initialize A as lower triangular, rest filled with -999
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);

		if (k <= i)
			A[state] = (num_t)((i + k) % 100) / m;
		else
			A[state] = (num_t)-999;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// C: i x j
	// A: i x k (symmetric, lower stored)
	// B: i x j
	#pragma scop
	traverser(C, A, B).template for_dims<'i'>([=](auto trav_i) {
		// loop over i
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			// loop over j
			num_t temp2 = 0;

			auto state_ij = trav_ij.state();
			auto i_idx = get_index<'i'>(state_ij);
			auto j_idx = get_index<'j'>(state_ij);

			// inner loop over k (we will skip k >= i)
			trav_ij.template for_dims<'k'>([&](auto trav_ijk) {
				auto state_ijk = trav_ijk.state();
				auto k_idx = get_index<'k'>(state_ijk);

				if (k_idx < i_idx) {
					// States for the required accesses:
					// C[k][j], B[i][j], B[k][j], A[i][k]
					auto s_kj = idx<'i', 'j'>(k_idx, j_idx);
					auto s_ij_local = state_ij; // i, j fixed
					auto s_ik = idx<'i', 'k'>(i_idx, k_idx);

					C[s_kj] += alpha * B[s_ij_local] * A[s_ik];
					temp2   +=          B[s_kj]      * A[s_ik];
				}
			});

			// Final update of C[i][j]
			auto s_ij = state_ij;
			auto a_ii = A[idx<'i', 'k'>(i_idx, i_idx)];

			C[s_ij] = beta * C[s_ij]
				+ alpha * B[s_ij] * a_ii
				+ alpha * temp2;
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t m = M;
	std::size_t n = N;

	// input data
	num_t alpha;
	num_t beta;

	// lengths for all dimensions
	auto set_lengths =
		noarr::set_length<'i'>(m)
		^ noarr::set_length<'j'>(n)
		^ noarr::set_length<'k'>(m);

	// data containers
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_symm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}