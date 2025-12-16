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
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(q_layout, j_vec ^ i_vec);

	// R: N x N (k x j)
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
// This implementation keeps the algorithm identical to the original
// but rewrites all loop nests using Noarr traversers.  The key ideas:
//
//  - Use column-major A and Q: fixing 'j' gives contiguous A[:, j], Q[:, j].
//  - Drive the outer k-loop by traversing the 'k' dimension of R.
//  - For each k, create column views A[:,k], Q[:,k] using noarr::fix<'j'>.
//  - Use noarr::slice<'j'> to restrict work to trailing columns j > k.
//  - Use traverser(...).for_dims<'j'> to iterate those columns, and
//    inner for_each to iterate rows 'i'.
//  - Replace per-element division by a single reciprocal per column.
//
// All iteration order and dependence constraints of modified Gram–Schmidt
// are preserved: k remains sequential, and for fixed k, each column j>k
// is processed independently.
[[gnu::flatten, gnu::noinline]]
void kernel_gramschmidt(auto A, auto R, auto Q) {
	using namespace noarr;

	auto m = A | get_length<'i'>();
	auto n = A | get_length<'j'>();

	(void)m; // m is not explicitly needed, traversal takes care of bounds

	#pragma scop
	// Outer loop over k implemented as a traversal over the 'k' dimension
	// of R. This keeps the k-iterations sequential but expressed via Noarr.
	traverser(R).template for_dims<'k'>([&](auto trav_k) {
		// Extract the current k index from the traverser's fixed state
		auto state_k = trav_k.state();
		std::size_t k = get_index<'k'>(state_k);

		// Column views of A and Q for the current k:
		// A_col_k: A[:, k], Q_col_k: Q[:, k]
		auto A_col_k = A ^ fix<'j'>(k);
		auto Q_col_k = Q ^ fix<'j'>(k);

		// ------------------------------------------------------------------
		// 1) Compute squared 2-norm of column k of A:
		//      nrm = sum_i A[i,k] * A[i,k]
		// ------------------------------------------------------------------
		num_t nrm = (num_t)0.0;

		traverser(A_col_k).for_each([&](auto state_i) {
			num_t a_ik = A_col_k[state_i];
			nrm += a_ik * a_ik;
		});

		// ------------------------------------------------------------------
		// 2) Normalize column k into Q[:,k] and set R[k,k]
		//
		//    R[k,k] = sqrt(nrm);
		//    Q[i,k] = A[i,k] / R[k,k];
		//
		// We compute inv_r_kk = 1 / R[k,k] once and use multiplication
		// instead of a division per element.
		// ------------------------------------------------------------------
		num_t r_kk = std::sqrt(nrm);
		R[idx<'k', 'j'>(k, k)] = r_kk;

		// Strength reduction: one division per column instead of M divisions.
		num_t inv_r_kk = (num_t)1.0 / r_kk;

		traverser(A_col_k, Q_col_k).for_each([&](auto state_i) {
			Q_col_k[state_i] = A_col_k[state_i] * inv_r_kk;
		});

		// ------------------------------------------------------------------
		// 3) Orthogonalize remaining columns j > k:
		//
		//    For each j in (k+1 .. n-1):
		//      R[k,j] = sum_i Q[i,k] * A[i,j]
		//      A[i,j] = A[i,j] - Q[i,k] * R[k,j]
		//
		// We express the triangular domain j>k by slicing the 'j' dimension.
		// ------------------------------------------------------------------

		// If there are no trailing columns, this k-iteration is done.
		if (k + 1 >= n)
			return;

		std::size_t j_start = k + 1;
		std::size_t trailing_len = n - j_start;

		// Trailing-column views:
		//   A_trail:  A[:, j_start .. n)
		//   R_trail:  R[k, j_start .. n)  (row k, restricted columns)
		auto A_trail = A ^ slice<'j'>(j_start, trailing_len);
		auto R_row_k = R ^ fix<'k'>(k);
		auto R_trail_k = R_row_k ^ slice<'j'>(j_start, trailing_len);

		// For each trailing column j, process:
		//   - r_kj = dot(Q[:,k], A[:,j])
		//   - R[k,j] = r_kj
		//   - A[:,j] -= Q[:,k] * r_kj
		//
		// We use a traverser over (A_trail, Q_col_k, R_trail_k) and
		// iterate the 'j' dimension explicitly via for_dims<'j'>. Inside
		// each j-tile, we traverse the remaining 'i' dimension with
	//	for_each, which yields contiguous column-major access.
		traverser(A_trail, Q_col_k, R_trail_k).template for_dims<'j'>([&](auto trav_j) {
			// Dot product for this column: r_kj = sum_i Q[i,k] * A[i,j]
			num_t r_kj = (num_t)0.0;

			trav_j.for_each([&](auto state_ij) {
				r_kj += Q_col_k[state_ij] * A_trail[state_ij];
			});

			// Store R[k,j] for this column.
			// `trav_j` acts as a state with the fixed 'j' index; R_trail_k
			// ignores any extra indices.
			R_trail_k[trav_j] = r_kj;

			// Rank-1 update of A[:,j]: A[i,j] -= Q[i,k] * R[k,j]
			trav_j.for_each([&](auto state_ij) {
				A_trail[state_ij] -= Q_col_k[state_ij] * r_kj;
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