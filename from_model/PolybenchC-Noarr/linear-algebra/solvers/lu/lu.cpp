#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "lu.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();
constexpr auto t_vec = noarr::vector<'t'>();

struct tuning {
	// A: i x j
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// 1D helper structures for k- and t-iterations (no actual data, only for traversal)
constexpr auto k_structure = noarr::scalar<char>() ^ k_vec;
constexpr auto t_structure = noarr::scalar<char>() ^ t_vec;

// initialization function
void init_array(auto A) {
	// A: i x j
	using namespace noarr;

	auto n = A | get_length<'i'>(); // matrix is n x n

	// Initialize A as unit-lower plus some values below diagonal, 0 above
	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		const int nn = static_cast<int>(n);
		const int jj = static_cast<int>(j);

		if (j < i) {
			// A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
			num_t val = static_cast<num_t>((-jj % nn));
			val /= static_cast<num_t>(nn);
			val += static_cast<num_t>(1);
			A[state] = val;
		} else if (j == i) {
			A[state] = static_cast<num_t>(1);
		} else { // j > i
			A[state] = static_cast<num_t>(0);
		}
	});

	// Make the matrix positive semi-definite:
	// B[r][s] = sum_t A[r][t] * A[s][t];  then A = B
	auto B = noarr::bag(A.structure());

	// Zero B
	traverser(B).for_each([&](auto state) {
		B[state] = static_cast<num_t>(0);
	});

	// Outer loop over t
	traverser(t_structure)
		.order(noarr::set_length<'t'>(n))
		.for_each([&](auto t_state) {
			auto t = get_index<'t'>(t_state);

			// Inner loops over r, s (i, j)
			traverser(B).for_each([&](auto rs_state) {
				auto [r, s] = get_indices<'i', 'j'>(rs_state);

				auto rt_state = noarr::idx<'i', 'j'>(r, t);
				auto st_state = noarr::idx<'i', 'j'>(s, t);

				B[rs_state] += A[rt_state] * A[st_state];
			});
		});

	// Copy B back into A
	traverser(A, B).for_each([&](auto state) {
		A[state] = B[state];
	});
}

// computation kernel
//
// This kernel implements an in-place LU factorization without pivoting.
// Compared to the original version, we apply two main optimizations while
// preserving the mathematical computation:
//
// 1) For the lower-triangular part (j < i), we form an explicit dot-product
//    accumulation over k and update A[i,j] once:
//          sum = sum_{k<j} A[i,k] * A[k,j]
//          A[i,j] = (A[i,j] - sum) / A[j,j]
//    This reduces repeated writes to A[i,j] and exposes a clear reduction
//    over k, improving ILP and giving the compiler a better chance to
//    vectorize the k-loop.
//
// 2) For the upper-triangular/diagonal part (j >= i), we interchange the
//    logical order of (j,k) loops. Instead of:
//          for j in [i, n):
//            for k in [0, i):
//              A[i,j] -= A[i,k] * A[k,j];
//    we implement a rank‑1 style update:
//          for k in [0, i):
//            aik = A[i,k];
//            for j in [i, n):
//              A[i,j] -= aik * A[k,j];
//    The sequence of k contributing to each (i,j) is unchanged, but we now
//    traverse j (the contiguous dimension) in the innermost loop and reuse
//    A[i,k] across all columns. This improves data locality and favors
//    SIMD/codegen, while the reduction over k remains mathematically
//    equivalent up to FP round-off ordering differences.
//
[[gnu::flatten, gnu::noinline]]
void kernel_lu(auto A) {
	// A: i x j (n x n)
	using namespace noarr;

	auto n = A | get_length<'i'>();

	#pragma scop
	traverser(A).template for_dims<'i'>([&](auto trav_i) {
		// trav_i has 'i' fixed, remaining dimension is 'j'
		auto state_i = trav_i.state();
		auto i_idx = get_index<'i'>(state_i);

		// ------------------------------------------------------------------
		// 1) Lower-triangular update: compute L(i,j) for 0 <= j < i
		//
		// Original code (schematically):
		//   for j in [0, i):
		//     for k in [0, j):
		//       A[i,j] -= A[i,k] * A[k,j];
		//     A[i,j] /= A[j,j];
		//
		// We keep the dependence structure in i and j, but change the
		// inner k-loop into an explicit reduction:
		//   sum = sum_{k<j} A[i,k]*A[k,j];
		//   A[i,j] = (A[i,j] - sum) / A[j,j];
		// ------------------------------------------------------------------
		if (i_idx > 0) {
			// Restrict j to [0, i_idx) for this row.
			auto trav_j_lt_i = trav_i.order(noarr::slice<'j'>(0, i_idx));

			trav_j_lt_i.template for_dims<'j'>([&](auto trav_ij) {
				auto state_ij = trav_ij.state();
				auto j_idx = get_index<'j'>(state_ij);

				// State for A[i,j]
				auto ij_state = noarr::idx<'i', 'j'>(i_idx, j_idx);

				// Compute sum_{k=0}^{j-1} A[i,k] * A[k,j]
				num_t sum = static_cast<num_t>(0);

				if (j_idx > 0) {
					traverser(k_structure)
						.order(noarr::set_length<'k'>(j_idx))
						.for_each([&](auto k_state) {
							auto k_idx = get_index<'k'>(k_state);

							auto ik_state = noarr::idx<'i', 'j'>(i_idx, k_idx);
							auto kj_state = noarr::idx<'i', 'j'>(k_idx, j_idx);

							sum += A[ik_state] * A[kj_state];
						});
				}

				// Divide by diagonal pivot A[j,j]
				auto jj_state = noarr::idx<'i', 'j'>(j_idx, j_idx);
				A[ij_state] = (A[ij_state] - sum) / A[jj_state];
			});
		}

		// ------------------------------------------------------------------
		// 2) Upper-triangular (and diagonal) update: compute U(i,j) for
		//    i <= j < n
		//
		// Original structure per row:
		//   for j in [i, n):
		//     for k in [0, i):
		//       A[i,j] -= A[i,k] * A[k,j];
		//
		// We apply a legal loop interchange in (j,k) to get:
		//   for k in [0, i):
		//     aik = A[i,k];
		//     for j in [i, n):
		//       A[i,j] -= aik * A[k,j];
		//
		// For each fixed (i,j) the set of k and their order remain the
		// same; only the nesting of loops changes. This improves cache
		// locality (j is contiguous in memory) and allows better reuse of
		// A[i,k] and A[k,*] across the row.
		// ------------------------------------------------------------------
		auto len_second = n - i_idx;
		if (len_second > 0) {
			// View of columns j in [i_idx, n) for this row i_idx.
			auto trav_j_ge_i = trav_i.order(noarr::slice<'j'>(i_idx, len_second));

			// For i_idx == 0, the reduction range [0, i_idx) is empty and
			// the row already contains the correct U(0, j) values.
			if (i_idx > 0) {
				// Outer loop over k: contributions from previous columns
				traverser(k_structure)
					.order(noarr::set_length<'k'>(i_idx))
					.for_each([&](auto k_state) {
						auto k_idx = get_index<'k'>(k_state);

						// A[i,k] reused across the whole row update for this k
						auto ik_state = noarr::idx<'i', 'j'>(i_idx, k_idx);
						const num_t aik = A[ik_state];

						// Inner loop over contiguous columns j >= i
						trav_j_ge_i.for_each([&](auto state_ij) {
							auto j_idx = get_index<'j'>(state_ij);

							auto kj_state = noarr::idx<'i', 'j'>(k_idx, j_idx);

							A[state_ij] -= aik * A[kj_state];
						});
					});
			}
		}
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// A: n x n
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_lu(A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (and prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}