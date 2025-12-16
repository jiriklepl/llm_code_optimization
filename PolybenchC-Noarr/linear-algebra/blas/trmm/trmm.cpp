#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "trmm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// A: i x k  (M x M)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: i x j  (M x N)
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;

// Array initialization
void init_array(num_t &alpha, auto A, auto B) {
	using namespace noarr;

	// A: i x k, square M x M
	// B: i x j, M x N

	alpha = (num_t)1.5;

	const int m_len = static_cast<int>(A | get_length<'i'>());
	const int n_len = static_cast<int>(B | get_length<'j'>());

	// Initialize A: unit lower-triangular
	// for (i = 0; i < m; i++) {
	//   for (j = 0; j < i; j++) A[i][j] = ((i + j) % m) / m;
	//   A[i][i] = 1.0;
	// }
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		int ii = static_cast<int>(i);
		int kk = static_cast<int>(k);

		if (kk < ii) {
			A[state] = (num_t)((ii + kk) % m_len) / (num_t)m_len;
		} else if (kk == ii) {
			A[state] = (num_t)1.0;
		}
		// For kk > ii we leave A[state] uninitialized, matching the C code
	});

	// Initialize B
	// for (i = 0; i < m; i++)
	//   for (j = 0; j < n; j++)
	//     B[i][j] = ((n + (i - j)) % n) / n;
	traverser(B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		int ii = static_cast<int>(i);
		int jj = static_cast<int>(j);

		B[state] = (num_t)(((n_len + (ii - jj)) % n_len)) / (num_t)n_len;
	});
}

// Main computational kernel
// Original C:
//
// for (i = 0; i < _PB_M; i++)
//   for (j = 0; j < _PB_N; j++) {
//     for (k = i+1; k < _PB_M; k++)
//       B[i][j] += A[k][i] * B[k][j];
//     B[i][j] = alpha * B[i][j];
//   }
//
// A is MxM (i x k), B is MxN (i x j)
[[gnu::flatten, gnu::noinline]]
void kernel_trmm(num_t alpha, auto A, auto B) {
	using namespace noarr;

	const std::size_t m_len = A | get_length<'i'>();

	#pragma scop
	traverser(B).template for_dims<'i'>([=](auto trav_i) {
		// trav_i has 'i' fixed, iterates over 'j'
		trav_i.template for_each<'j'>([=](auto state) {
			auto [i, j] = get_indices<'i', 'j'>(state);

			num_t Bij = B[state];

			// k-loop: for (k = i+1; k < M; k++)
			for (std::size_t k = i + 1; k < m_len; ++k) {
				// A[k][i] -> A at (i = k, k = i)
				auto a_state = noarr::idx<'i', 'k'>(k, i);
				// B[k][j] -> B at (i = k, j = j)
				auto bkj_state = noarr::idx<'i', 'j'>(k, j);

				Bij += A[a_state] * B[bkj_state];
			}

			Bij *= alpha;
			B[state] = Bij;
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

	// set lengths for all dimensions used
	auto set_lengths = noarr::set_length<'i'>(m)
	                 ^ noarr::set_length<'j'>(n)
	                 ^ noarr::set_length<'k'>(m);

	// allocate bags
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths); // M x M
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths); // M x N

	// initialize data
	init_array(alpha, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trmm(alpha, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (for DCE prevention / correctness check)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, B.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}