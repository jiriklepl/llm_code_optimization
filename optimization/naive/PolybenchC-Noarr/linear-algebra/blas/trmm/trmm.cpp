#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures_extended.hpp> // for fix, shift, index_in, etc.
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
//
// Optimized version:
//
// We keep `i` as the outer loop (to preserve the in-place dependency pattern),
// but we reorder the inner loops to (k, j) and use Noarr views to:
//   - iterate k > i using a shifted column view of A
//   - update whole rows of B with good spatial locality
//
// New logical order:
//
// for i = 0 .. M-1
//   for k = i+1 .. M-1
//     aik = A[k][i]
//     for j = 0 .. N-1
//       B[i][j] += aik * B[k][j];
//   for j = 0 .. N-1
//     B[i][j] *= alpha;
//
// This is algebraically equivalent to the original kernel and preserves
// the requirement that every use of B[k][j] as a multiplicand sees the
// original value, not yet updated.
[[gnu::flatten, gnu::noinline]]
void kernel_trmm(num_t alpha, auto A, auto B) {
	using namespace noarr;

	#pragma scop
	// Outer loop over rows i of B (and A)
	traverser(B).template for_dims<'i'>([&](auto trav_i) {
		// trav_i has 'i' fixed, iterates over 'j'
		auto i_state = trav_i.state();
		auto i_idx = get_index<'i'>(i_state); // may be static or dynamic
		std::size_t i = static_cast<std::size_t>(i_idx);

		// Build a view of column i of A, but only for rows k > i:
		//
		//   - fix<'k'>(i):   select column i     -> dims: 'i' (row index)
		//   - shift<'i'>(i+1): skip rows 0..i, so new index 0 corresponds
		//                      to original row (i+1)
		//
		// A_col_tail has a single dimension 'i' whose indices enumerate
		// k = i+1, i+2, ..., M-1 in order.
		auto A_col_tail = A
			^ fix<'k'>(i)      // fix column index = i
			^ shift<'i'>(i + 1u); // start from row i+1

		// Iterate over all k > i (rows below the diagonal)
		traverser(A_col_tail).for_each([&](auto k_state_tail) {
			// Logical row index k in the original A/B
			auto k_local = get_index<'i'>(k_state_tail); // 0 .. (#rows_tail-1)
			std::size_t k = static_cast<std::size_t>(k_local) + i + 1u;

			// A[k][i]: current element in the strictly lower triangular part
			num_t aik = A_col_tail[k_state_tail];

			// For this (i,k) pair, update the whole row i of B:
			//   B[i][j] += A[k][i] * B[k][j]   for all j
			//
			// We iterate over j using trav_i (which has 'i' fixed).
			trav_i.for_each([&](auto ij_state) {
				// Destination element B[i][j]
				num_t &Bij = B[ij_state];

				// Source element B[k][j]: keep 'j' from ij_state and
				// replace 'i' with k.
				auto kj_state =
					ij_state.template with<index_in<'i'>>(k);

				Bij += aik * B[kj_state];
			});
		});

		// Finally, scale the entire row i of B by alpha:
		trav_i.for_each([&](auto ij_state) {
			B[ij_state] *= alpha;
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