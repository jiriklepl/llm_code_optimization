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

	// ------------------------------------------------------------------
	// Step 1: Initialize A as unit-lower + some values below diagonal, 0 above
	// ------------------------------------------------------------------
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

	// ------------------------------------------------------------------
	// Step 2: Make the matrix positive semi-definite:
	//         B[r][s] = sum_t A[r][t] * A[s][t];  then A = B
	//
	// The original version did:
	//   for t
	//     for r,s    (full 2D traverser on every t)
	//       B[r,s] += A[r,t] * A[s,t];
	//
	// Here we keep the same (t, r, s) iteration order, but we:
	//   - traverse B with for_dims<'i','j'> (explicit row/col loops)
	//   - load A[r,t] once per (t,r) and reuse it across all s
	// This improves cache use without changing the arithmetic order
	// of updates to each B[r,s], so results remain bit-identical.
	// ------------------------------------------------------------------
	auto B = noarr::bag(A.structure());

	// Zero B
	traverser(B).for_each([&](auto state) {
		B[state] = static_cast<num_t>(0);
	});

	// Reusable traverser for B to avoid reconstructing it for every t
	auto B_trav_base = traverser(B);

	// Outer loop over t
	traverser(t_structure)
		.order(noarr::set_length<'t'>(n))
		.for_each([&](auto t_state) {
			auto t = get_index<'t'>(t_state);

			// For fixed t, iterate rows r (dimension 'i')
			B_trav_base.for_dims<'i'>([&](auto trav_r) {
				auto state_r = trav_r.state();
				auto r = get_index<'i'>(state_r);

				// A[r][t] is reused for all s in this row r
				auto rt_state = noarr::idx<'i', 'j'>(r, t);
				num_t art = A[rt_state];

				// Now iterate columns s (dimension 'j') inside the same row r
				trav_r.template for_dims<'j'>([&](auto trav_rs) {
					auto rs_state = trav_rs.state();
					auto s = get_index<'j'>(rs_state);

					auto st_state = noarr::idx<'i', 'j'>(s, t);

					B[rs_state] += art * A[st_state];
				});
			});
		});

	// Copy B back into A
	traverser(A, B).for_each([&](auto state) {
		A[state] = B[state];
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_lu(auto A) {
	// A: i x j (n x n)
	using namespace noarr;

	auto n = A | get_length<'i'>();

	// Base traverser for the k-dimension. We reuse it and only apply
	// length changes via order(noarr::set_length<'k'>(...)) inside.
	auto k_trav_base = traverser(k_structure);

	#pragma scop
	// Outer loop over i (row index).
	traverser(A).template for_dims<'i'>([&](auto trav_i) {
		auto state_i = trav_i.state();
		auto i_idx = get_index<'i'>(state_i);

		// Single traversal over j; we branch on j<i vs j>=i instead of
		// constructing two different sliced traversers. This preserves
		// the original execution order:
		//   first j in [0, i), then j in [i, n),
		// but with less traversal overhead.
		trav_i.template for_dims<'j'>([&](auto trav_ij) {
			auto state_ij = trav_ij.state(); // contains both 'i' and 'j'
			auto j_idx = get_index<'j'>(state_ij);

			// Lower part: 0 <= j < i  -> compute L(i,j)
			if (j_idx < i_idx) {
				// A[i][j] -= sum_{k=0}^{j-1} A[i][k] * A[k][j]
				if (j_idx > 0) {
					// k in [0, j_idx)
					auto k_trav = k_trav_base.order(noarr::set_length<'k'>(j_idx));
					k_trav.for_each([&](auto k_state) {
						auto k_idx = get_index<'k'>(k_state);

						auto ik_state = noarr::idx<'i', 'j'>(i_idx, k_idx);
						auto kj_state = noarr::idx<'i', 'j'>(k_idx, j_idx);

						A[state_ij] -= A[ik_state] * A[kj_state];
					});
				}

				// A[i][j] /= A[j][j];
				auto jj_state = noarr::idx<'i', 'j'>(j_idx, j_idx);
				A[state_ij] /= A[jj_state];

			// Upper part: i <= j < n  -> compute U(i,j)
			} else {
				// A[i][j] -= sum_{k=0}^{i-1} A[i][k] * A[k][j]
				if (i_idx > 0) {
					// k in [0, i_idx)
					auto k_trav = k_trav_base.order(noarr::set_length<'k'>(i_idx));
					k_trav.for_each([&](auto k_state) {
						auto k_idx = get_index<'k'>(k_state);

						auto ik_state = noarr::idx<'i', 'j'>(i_idx, k_idx);
						auto kj_state = noarr::idx<'i', 'j'>(k_idx, j_idx);

						A[state_ij] -= A[ik_state] * A[kj_state];
					});
				}
			}
		});
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