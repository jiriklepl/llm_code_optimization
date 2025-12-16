#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions (DATA_TYPE, M, N, DEFINE_PROTO_STRUCT, noarr::bag, ...)
#include "defines.hpp"

// include benchmark-specific definitions (PolyBench sizes, etc.)
#include "gramschmidt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// A, Q:  M x N  (i x j)
	// Layout is chosen as (i_vec ^ j_vec), i.e. 'j' outer, 'i' inner.
	// Fixing a column (fix<'j'>(k)) then iterating over 'i' walks contiguous
	// memory, which matches the column-oriented Gram-Schmidt algorithm.
	DEFINE_PROTO_STRUCT(a_layout, i_vec ^ j_vec);
	DEFINE_PROTO_STRUCT(q_layout, i_vec ^ j_vec);

	// R: N x N (k x j)
	// Keep the original row-major style layout for R: 'k' outer, 'j' inner,
	// so that iterating j for a fixed k writes a contiguous row.
	DEFINE_PROTO_STRUCT(r_layout, j_vec ^ k_vec);
} tuning;


// Array initialization --------------------------------------------------------
void init_array(auto A, auto R, auto Q) {
	using namespace noarr;

	// Dimensions taken from the structures
	auto m = A | get_length<'i'>();
	auto n = A | get_length<'j'>();

	// Initialize A and Q
	// for (i = 0; i < m; i++)
	//   for (j = 0; j < n; j++) {
	//     A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
	//     Q[i][j] = 0.0;
	//   }
	traverser(A, Q).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		A[state] = (((num_t)((i * j) % m) / (num_t)m) * (num_t)100.0) + (num_t)10.0;
		Q[state] = (num_t)0.0;
	});

	// Initialize R
	// for (i = 0; i < n; i++)
	//   for (j = 0; j < n; j++)
	//     R[i][j] = 0.0;
	traverser(R).for_each([=](auto state) {
		R[state] = (num_t)0.0;
	});
}


// Main computational kernel --------------------------------------------------
// QR Decomposition with Modified Gram-Schmidt
//
// The entire loop nest over k, j, i is expressed using Noarr traversers:
//  - an artificial 1D structure over 'k' drives the outer iteration,
//  - another 1D structure over 'j' plus a slice implements j = k+1..n-1,
//  - column views A[:,k], Q[:,k], A[:,j] are created with fix<'j'>(...),
//    and traversed over 'i' only.
// With the tuned layouts, these column traversals are contiguous in memory.
[[gnu::flatten, gnu::noinline]]
void kernel_gramschmidt(auto A, auto R, auto Q) {
	using namespace noarr;

	auto m = A | get_length<'i'>();
	auto n = A | get_length<'j'>();

	(void)m; // m is not explicitly needed; traversal takes care of bounds

	// Synthetic 1D iteration spaces for 'k' and 'j', used only to obtain states.
	auto k_space = noarr::scalar<char>() ^ noarr::vector<'k'>(n);
	auto j_space = noarr::scalar<char>() ^ noarr::vector<'j'>(n);

	// Pre-construct traversers for the synthetic spaces; we will further
	// transform them (via order(slice)) inside the loop.
	auto k_trav = noarr::traverser(k_space);
	auto j_trav = noarr::traverser(j_space);

	#pragma scop
	k_trav.for_each([&](auto s_k) {
		const std::size_t k = noarr::get_index<'k'>(s_k);

		// nrm = sum_i A[i][k]^2
		num_t nrm = (num_t)0.0;

		// k-th column views of A and Q (A[:, k], Q[:, k]).
		// With layout (i_vec ^ j_vec), fixing 'j' gives a 1D structure over 'i'
		// whose elements are contiguous in memory.
		auto A_col_k = A ^ noarr::fix<'j'>(k);
		auto Q_col_k = Q ^ noarr::fix<'j'>(k);

		noarr::traverser(A_col_k).for_each([&](auto s_i) {
			const num_t a_ik = A_col_k[s_i];
			nrm += a_ik * a_ik;
		});

		// R[k][k] = sqrt(nrm);
		const num_t r_kk = std::sqrt(nrm);
		R[noarr::idx<'k', 'j'>(k, k)] = r_kk;

		// Q[i][k] = A[i][k] / R[k][k];
		noarr::traverser(A_col_k, Q_col_k).for_each([&](auto s_i) {
			Q_col_k[s_i] = A_col_k[s_i] / r_kk;
		});

		// Triangular j-loop: for (j = k + 1; j < n; j++) { ... }
		// Implemented as a traversal over the 1D 'j' space restricted by slice.
		j_trav
			.order(noarr::slice<'j'>(k + 1, n - (k + 1))) // j in [k+1, n)
			.for_each([&](auto s_j) {
				const std::size_t j = noarr::get_index<'j'>(s_j);

				// R[k][j] = sum_i Q[i][k] * A[i][j];
				num_t r_kj = (num_t)0.0;

				// j-th column view of A: A[:, j]
				auto A_col_j = A ^ noarr::fix<'j'>(j);

				noarr::traverser(Q_col_k, A_col_j).for_each([&](auto s_i) {
					r_kj += Q_col_k[s_i] * A_col_j[s_i];
				});

				R[noarr::idx<'k', 'j'>(k, j)] = r_kj;

				// A[i][j] = A[i][j] - Q[i][k] * R[k][j];
				noarr::traverser(Q_col_k, A_col_j).for_each([&](auto s_i) {
					A_col_j[s_i] = A_col_j[s_i] - Q_col_k[s_i] * r_kj;
				});
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

	// set lengths for all dimensions used
	auto set_lengths =
		noarr::set_length<'i'>(m) ^
		noarr::set_length<'j'>(n) ^
		noarr::set_length<'k'>(n);

	// allocate bags
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto R = noarr::bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ set_lengths);
	auto Q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), R.get_ref(), Q.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gramschmidt(A.get_ref(), R.get_ref(), Q.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// optional: print results to stderr (R then Q) to prevent dead-code elimination
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, R.get_ref() ^ noarr::hoist<'k'>());
		noarr::serialize_data(std::cerr, Q.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing to stdout
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}