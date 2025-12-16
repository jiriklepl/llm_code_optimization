#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions (DATA_TYPE, N, M, DEFINE_PROTO_STRUCT, ...)
#include "defines.hpp"

// include benchmark-specific definitions (N, M, DATA_TYPE, etc.)
#include "syr2k.hpp"

using num_t = DATA_TYPE;

namespace {

// basic dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

// layout tuning (can be changed independently of the algorithm)
struct tuning {
	// C is N x N, indexed by (i, j)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // physical layout: j (outer) x i (inner)

	// A is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // physical layout: k (outer) x i (inner)

	// B is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ i_vec); // physical layout: k (outer) x i (inner)
} tuning;

// initialization function
// C: i x j  (N x N)
// A: i x k  (N x M)
// B: i x k  (N x M)
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	// Initialize A: A[i][k] = ( (i * k + 1) % n ) / n
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto n_len  = A | get_length<'i'>(); // corresponds to original 'n'
		A[state] = (num_t)((i * k + 1) % n_len) / n_len;
	});

	// Initialize B: B[i][k] = ( (i * k + 2) % m ) / m
	traverser(B).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto m_len  = B | get_length<'k'>(); // corresponds to original 'm'
		B[state] = (num_t)((i * k + 2) % m_len) / m_len;
	});

	// Initialize C: C[i][j] = ( (i * j + 3) % n ) / m
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto n_len  = C | get_length<'i'>(); // N
		auto m_len  = B | get_length<'k'>(); // M
		C[state] = (num_t)((i * j + 3) % n_len) / m_len;
	});
}

// computation kernel
// BLAS-like SYR2K:
//
// C := alpha * A * B^T + alpha * B * A^T + beta * C
//
// with:
//   A: N x M
//   B: N x M
//   C: N x N (only lower-triangular part (i,j) with j <= i is updated)
[[gnu::flatten, gnu::noinline]]
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// C: i x j
	// A: i x k
	// B: i x k

#pragma scop
	// Outer loop over i (rows of C)
	traverser(C, A, B).template for_dims<'i'>([=](auto trav_i) {
		// get the current i index
		auto state_i = trav_i.state();
		auto i_idx   = get_index<'i'>(state_i);

		// ---------------------------------------------------------------------
		// First loop nest:
		//   for (j = 0; j <= i; j++)
		//       C[i][j] *= beta;
		// ---------------------------------------------------------------------
		trav_i.template for_each<'j'>([=](auto state) {
			auto i_cur = get_index<'i'>(state);
			auto j_cur = get_index<'j'>(state);

			if (j_cur <= i_cur) {
				// state contains indices (i, j) (and possibly others ignored by C)
				C[state] *= beta;
			}
		});

		// ---------------------------------------------------------------------
		// Second loop nest:
		//   for (k = 0; k < M; k++)
		//     for (j = 0; j <= i; j++)
		//       C[i][j] += A[j][k] * alpha * B[i][k]
		//                  + B[j][k] * alpha * A[i][k];
		// ---------------------------------------------------------------------
		trav_i.template for_dims<'k'>([=](auto trav_k) {
			// k is fixed here, i is also fixed from the outer for_dims<'i'>

			trav_k.template for_each<'j'>([=](auto state) {
				// state has indices (i, j, k)
				auto i_cur = get_index<'i'>(state);
				auto j_cur = get_index<'j'>(state);

				if (j_cur <= i_cur) {
					// Build a state where the row index for A/B is j instead of i:
					auto state_jk = state.template with<index_in<'i'>>(j_cur);

					// A[j][k], B[j][k]
					num_t A_jk = A[state_jk];
					num_t B_jk = B[state_jk];

					// A[i][k], B[i][k]
					num_t A_ik = A[state];
					num_t B_ik = B[state];

					// Update C[i][j]
					C[state] += A_jk * alpha * B_ik + B_jk * alpha * A_ik;
				}
			});
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t m = M;

	// scalar parameters
	num_t alpha;
	num_t beta;

	// set lengths for all relevant dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n) ^
		noarr::set_length<'k'>(m);

	// allocate bags for C, A, B with selected layouts and lengths
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syr2k(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (only when some argument is given, mimicking PolyBench's DCE guard)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'i' to make the output row-major in (i, j)
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}