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

	// Initialize x, y, b
	// for (i = 0; i < n; i++) {
	//   x[i] = 0;
	//   y[i] = 0;
	//   b[i] = (i+1)/fn/2.0 + 4;
	// }
	traverser(b, x, y).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x[state] = num_t(0);
		y[state] = num_t(0);

		x[state]; // keep structure usage symmetrical
		y[state];

		b[state] = (static_cast<num_t>(i + 1) / fn / num_t(2.0)) + num_t(4);
	});

	// Initialize A:
	// for (i = 0; i < n; i++) {
	//   for (j = 0; j <= i; j++)
	//     A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
	//   for (j = i+1; j < n; j++)
	//     A[i][j] = 0;
	//   A[i][i] = 1;
	// }
	// Rewritten as a single 2D traversal with case distinction on j vs i.
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

	// Make the matrix positive semi-definite:
	// B[r][s] = 0
	// for (t = 0; t < n; ++t)
	//   for (r = 0; r < n; ++r)
	//     for (s = 0; s < n; ++s)
	//       B[r][s] += A[r][t] * A[s][t];
	// A[r][s] = B[r][s]

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
[[gnu::flatten, gnu::noinline]]
void kernel_ludcmp(auto A, auto b, auto x, auto y) {
	using namespace noarr;

	// Length in the 'i' dimension
	std::size_t n = A | get_length<'i'>();

	// LU factorization (Doolittle-like scheme)
	//
	// Original:
	// for (i = 0; i < N; i++) {
	//   for (j = 0; j < i; j++) {
	//     w = A[i][j];
	//     for (k = 0; k < j; k++)
	//       w -= A[i][k] * A[k][j];
	//     A[i][j] = w / A[j][j];
	//   }
	//   for (j = i; j < N; j++) {
	//     w = A[i][j];
	//     for (k = 0; k < i; k++)
	//       w -= A[i][k] * A[k][j];
	//     A[i][j] = w;
	//   }
	// }
	//
	// Rewritten using a single j-loop per i:
	//   max_k = (j < i ? j : i);
	//   w = A[i][j] - sum_{k=0..max_k-1} A[i][k]*A[k][j];
	//   if (j < i) A[i][j] = w / A[j][j];
	//   else       A[i][j] = w;

	traverser(A).template for_dims<'i'>([&](auto trav_i) {
		auto s_i = trav_i.state();
		std::size_t i_idx = get_index<'i'>(s_i);

		trav_i.template for_each<'j'>([&](auto state) {
			auto j_idx = get_index<'j'>(state);

			num_t w = A[state];

			std::size_t max_k = (j_idx < i_idx) ? j_idx : i_idx;
			for (std::size_t k = 0; k < max_k; ++k) {
				w -= A.template at<'i', 'j'>(i_idx, k) *
				     A.template at<'i', 'j'>(k, j_idx);
			}

			if (j_idx < i_idx) {
				A[state] = w / A.template at<'i', 'j'>(j_idx, j_idx);
			} else {
				A[state] = w;
			}
		});
	});

	// Forward substitution: solve Ly = b
	//
	// for (i = 0; i < N; i++) {
	//   w = b[i];
	//   for (j = 0; j < i; j++)
	//     w -= A[i][j] * y[j];
	//   y[i] = w;
	// }

	traverser(y, b, A).template for_dims<'i'>([&](auto trav_i) {
		auto s_i = trav_i.state();
		std::size_t i_idx = get_index<'i'>(s_i);

		num_t w = b[s_i];

		trav_i.template for_each<'j'>([&](auto state) {
			auto j_idx = get_index<'j'>(state);
			if (j_idx < i_idx) {
				w -= A.template at<'i', 'j'>(i_idx, j_idx) *
				     y.template at<'i'>(j_idx);
			}
		});

		y[s_i] = w;
	});

	// Backward substitution: solve Ux = y
	//
	// for (i = N-1; i >= 0; i--) {
	//   w = y[i];
	//   for (j = i+1; j < N; j++)
	//     w -= A[i][j] * x[j];
	//   x[i] = w / A[i][i];
	// }
	//
	// Implemented with an ascending traversal over an "inverse" index:
	//   inv = 0..N-1, i = N-1-inv

	traverser(x, y, A).template for_dims<'i'>([&](auto trav_i) {
		auto s_inv = trav_i.state();
		std::size_t inv = get_index<'i'>(s_inv);
		std::size_t i_idx = n - 1 - inv;

		num_t w = y.template at<'i'>(i_idx);

		trav_i.template for_each<'j'>([&](auto state) {
			auto j_idx = get_index<'j'>(state);
			if (j_idx > i_idx) {
				w -= A.template at<'i', 'j'>(i_idx, j_idx) *
				     x.template at<'i'>(j_idx);
			}
		});

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