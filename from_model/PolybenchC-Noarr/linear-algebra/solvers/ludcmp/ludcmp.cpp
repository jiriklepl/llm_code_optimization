#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "ludcmp.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A: i x j
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	// 1D vectors (b, x, y): i
	DEFINE_PROTO_STRUCT(vec_layout, i_vec);
} tuning;

// Array initialization
void init_array(int n, auto A, auto b, auto x, auto y) {
	using namespace noarr;

	num_t fn = static_cast<num_t>(n);

	// ------------------------------------------------------------
	// Initialize x, y, b
	//   x[i] = 0
	//   y[i] = 0
	//   b[i] = (i+1)/fn/2.0 + 4
	// ------------------------------------------------------------
	traverser(b, x, y).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x[state] = num_t(0);
		y[state] = num_t(0);

		x[state]; // keep structure usage symmetrical
		y[state];

		b[state] = (static_cast<num_t>(i + 1) / fn / num_t(2.0)) + num_t(4);
	});

	// ------------------------------------------------------------
	// Initialize A:
	//
	// for (i = 0; i < n; i++) {
	//   for (j = 0; j <= i; j++)
	//     A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
	//   for (j = i+1; j < n; j++)
	//     A[i][j] = 0;
	//   A[i][i] = 1;
	// }
	//
	// Rewritten as a single 2D traversal with case distinction on j vs i.
	// ------------------------------------------------------------
	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		if (j < i) {
			// (DATA_TYPE)(-j % n) / n + 1;
			int nn = n;
			int jj = static_cast<int>(j);
			int rem = (-jj) % nn;
			A[state] = static_cast<num_t>(rem) / fn + num_t(1);
		} else if (j == i) {
			A[state] = num_t(1);
		} else {
			A[state] = num_t(0);
		}
	});

	// ------------------------------------------------------------
	// Make the matrix positive semi-definite:
	//
	// B[r][s] = 0
	// for (t = 0; t < n; ++t)
	//   for (r = 0; r < n; ++r)
	//     for (s = 0; s < n; ++s)
	//       B[r][s] += A[r][t] * A[s][t];
	// A[r][s] = B[r][s]
	//
	// Implemented with Noarr views and a triple traversal.
	// This phase is outside the timed kernel; we keep it structurally
	// identical to the original but expressed with Noarr.
	// ------------------------------------------------------------

	// B has the same structure as A
	auto B = noarr::bag(A.structure());

	// Zero out B
	traverser(B).for_each([&](auto state) {
		B[state] = num_t(0);
	});

	// Create views:
	// A_rt: A[r][t]
	auto A_rt = A.get_ref() ^ noarr::rename<'i', 'r', 'j', 't'>();
	// A_st: A[s][t]
	auto A_st = A.get_ref() ^ noarr::rename<'i', 's', 'j', 't'>();
	// B_rs: B[r][s]
	auto B_rs = B.get_ref() ^ noarr::rename<'i', 'r', 'j', 's'>();
	// B_rst: B[r][s] broadcast over t
	auto B_rst = B_rs ^ noarr::bcast<'t'>(n);

	// Accumulate B[r][s] += A[r][t] * A[s][t]
	// Original loop order was (t, r, s); we preserve that here.
	noarr::traverser(B_rst, A_rt, A_st)
		.order(noarr::hoist<'t'>())
		.for_each([&](auto state) {
			B_rst[state] += A_rt[state] * A_st[state];
		});

	// Copy B back into A: A[r][s] = B[r][s]
	noarr::traverser(A, B).for_each([&](auto state) {
		A[state] = B[state];
	});
}

// Main computational kernel
//
// This version restructures the loops to improve control-flow and
// data locality, while keeping the exact mathematical algorithm:
//
// 1) In-place LU factorization (Doolittle, no pivoting)
// 2) Forward substitution (Ly = b)
// 3) Backward substitution (Ux = y)
//
// Key Noarr-based optimizations in this kernel:
//
// - For each fixed row i, the j-domain is split using noarr::slice<'j'>
//   into the strictly lower (j < i) and upper (j >= i) parts.
//   This removes per-iteration branching on j and avoids traversing
//   irrelevant j-values.
//
// - All iteration over the matrix indices (i, j) is expressed via
//   Noarr traversers (for_dims + order(slice(...))). The only
//   remaining explicit for-loops are the inner reductions over k,
//   which are simple scalar reductions not tied to the layout.
//
[[gnu::flatten, gnu::noinline]]
void kernel_ludcmp(auto A, auto b, auto x, auto y) {
	using namespace noarr;

	// Matrix dimension (N x N)
	std::size_t n = A | get_length<'i'>();

	// ============================================================
	// 1. LU factorization in-place (Doolittle scheme)
	//
	// Classical form:
	//   for i = 0..N-1:
	//     for j = 0..i-1:
	//       A[i,j] = (A[i,j] - sum_{k=0..j-1} A[i,k]*A[k,j]) / A[j,j]
	//     for j = i..N-1:
	//       A[i,j] = A[i,j] - sum_{k=0..i-1} A[i,k]*A[k,j]
	//
	// Below, we:
	//   - traverse rows i using traverser(A).for_dims<'i'>
	//   - use order(slice<'j'>) to get the two j-ranges per row
	//     without any explicit branching on j
	//   - keep the inner reductions over k as explicit for-loops
	// ============================================================
	traverser(A).template for_dims<'i'>([&](auto trav_i) {
		// Fixed row index i for this iteration
		auto s_i = trav_i.state();
		std::size_t i_idx = get_index<'i'>(s_i);

		// -----------------------
		// 1.1 Strictly lower part
		//     j in [0, i_idx)
		// -----------------------
		//
		// Slice the j-dimension so that only j < i contribute.
		// For i_idx == 0, the slice length is zero and the traversal
		// is empty, as desired.
		auto lower_trav = trav_i.order(noarr::slice<'j'>(0, i_idx));

		lower_trav.for_each([&](auto state) {
			auto j_idx = get_index<'j'>(state); // 0 <= j_idx < i_idx

			// w = A[i][j] - sum_{k=0..j-1} A[i][k] * A[k][j]
			num_t w = A[state];

			for (std::size_t k = 0; k < j_idx; ++k) {
				w -= A.template at<'i', 'j'>(i_idx, k) *
				     A.template at<'i', 'j'>(k, j_idx);
			}

			// Divide by the pivot A[j][j]; store L(i,j) in-place
			w /= A.template at<'i', 'j'>(j_idx, j_idx);
			A[state] = w;
		});

		// -----------------------
		// 1.2 Upper part (including diagonal)
		//     j in [i_idx, n)
		// -----------------------
		//
		// Slice j so that only the upper triangle (and diagonal) is
		// visited. Each element in this range is:
		//   A[i,j] = A[i,j] - sum_{k=0..i-1} A[i,k] * A[k,j]
		//
		// For i_idx == n-1 this is a single element j == i_idx; for
		// larger i we skip earlier elements by construction.
		auto upper_trav = trav_i.order(noarr::slice<'j'>(i_idx, n - i_idx));

		upper_trav.for_each([&](auto state) {
			auto j_idx = get_index<'j'>(state); // i_idx <= j_idx < n

			num_t w = A[state];

			for (std::size_t k = 0; k < i_idx; ++k) {
				w -= A.template at<'i', 'j'>(i_idx, k) *
				     A.template at<'i', 'j'>(k, j_idx);
			}

			// Store U(i,j) in-place
			A[state] = w;
		});
	});

	// ============================================================
	// 2. Forward substitution: solve L y = b
	//
	//   for (i = 0; i < N; i++) {
	//     w = b[i];
	//     for (j = 0; j < i; j++)
	//       w -= A[i][j] * y[j];
	//     y[i] = w;
	//   }
	//
	// L is stored in the strict lower part of A with unit diagonal.
	//
	// We again:
	//   - traverse rows i with for_dims<'i'>
	//   - restrict j to [0, i) via slice<'j'>
	//   - keep the inner reduction as an explicit scalar loop
	// ============================================================
	traverser(y, b, A).template for_dims<'i'>([&](auto trav_i) {
		auto s_i = trav_i.state();
		std::size_t i_idx = get_index<'i'>(s_i);

		// Start with b[i]
		num_t w = b[s_i];

		// Only j < i contribute to the sum; slice the j-dimension.
		auto lower_trav = trav_i.order(noarr::slice<'j'>(0, i_idx));

		lower_trav.for_each([&](auto state) {
			auto j_idx = get_index<'j'>(state); // 0 <= j_idx < i_idx

			w -= A.template at<'i', 'j'>(i_idx, j_idx) *
			     y.template at<'i'>(j_idx);
		});

		y[s_i] = w;
	});

	// ============================================================
	// 3. Backward substitution: solve U x = y
	//
	// Classical descending form:
	//   for (i = N-1; i >= 0; i--) {
	//     w = y[i];
	//     for (j = i+1; j < N; j++)
	//       w -= A[i][j] * x[j];
	//     x[i] = w / A[i][i];
	//   }
	//
	// The original code expresses this using an ascending "inverse"
	// index inv = 0..N-1 with i = N-1-inv. We keep that mapping so
	// that the Noarr traverser can run in the natural 0..N-1 order
	// while still respecting the true data dependencies.
	//
	// As in the forward solve, we restrict the j-range per row using
	// a j-slice instead of an if-condition inside the loop.
	// ============================================================
	traverser(x, y, A).template for_dims<'i'>([&](auto trav_inv) {
		auto s_inv = trav_inv.state();
		std::size_t inv = get_index<'i'>(s_inv); // 0..N-1
		std::size_t i_idx = n - 1 - inv;        // actual row index (N-1..0)

		// Start with y[i]
		num_t w = y.template at<'i'>(i_idx);

		// Only j > i contribute to the reduction; if i_idx == n-1
		// the slice is empty and no iterations are run.
		if (i_idx + 1 < n) {
			auto upper_trav = trav_inv.order(
				noarr::slice<'j'>(i_idx + 1, n - (i_idx + 1))
			);

			upper_trav.for_each([&](auto state) {
				auto j_idx = get_index<'j'>(state); // i_idx+1 <= j_idx < n

				w -= A.template at<'i', 'j'>(i_idx, j_idx) *
				     x.template at<'i'>(j_idx);
			});
		}

		// Divide by the diagonal pivot U(i,i) = A[i][i]
		x.template at<'i'>(i_idx) =
			w / A.template at<'i', 'j'>(i_idx, i_idx);
	});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// Problem size
	std::size_t n = N;

	// Structures and bags
	auto set_len_i = noarr::set_length<'i'>(n);
	auto set_len_ij = set_len_i ^ noarr::set_length<'j'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_len_ij);
	auto b = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ set_len_i);
	auto x = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ set_len_i);
	auto y = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ set_len_i);

	// Initialize data
	init_array(static_cast<int>(n),
	           A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// Run kernel
	kernel_ludcmp(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// Output result to stderr (for DCE / checking)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x.get_ref());
	}

	// Output timing to stdout
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}