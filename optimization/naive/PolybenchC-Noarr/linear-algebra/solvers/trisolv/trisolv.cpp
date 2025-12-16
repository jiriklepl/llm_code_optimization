#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions (DATA_TYPE, N, DEFINE_PROTO_STRUCT, ...)
#include "defines.hpp"

// include benchmark-specific definitions (for trisolv)
#include "trisolv.hpp"

using num_t = DATA_TYPE;

namespace {

// dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

// layouts for individual arrays
struct tuning {
	// L: i x j  (row i, column j)
	DEFINE_PROTO_STRUCT(l_layout, j_vec ^ i_vec);
	// x: i
	DEFINE_PROTO_STRUCT(x_layout, i_vec);
	// b: i
	DEFINE_PROTO_STRUCT(b_layout, i_vec);
} tuning;

// Array initialization.
// L: n x n lower triangular
// x: length n
// b: length n
void init_array(auto L, auto x, auto b) {
	using namespace noarr;

	// problem size from structure
	auto n = L | get_length<'i'>();

	// initialize x and b: for i in [0, n)
	traverser(x, b).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x[state] = (num_t)-999;
		b[state] = (num_t)i;
	});

	// initialize L: for i in [0, n), for j in [0, n), but only j <= i are set
	traverser(L).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		if (j <= i) {
			L[state] = (num_t)(i + n - j + 1) * 2 / n;
		}
	});
}

// Main computational kernel:
// for i = 0..n-1:
//   x[i] = b[i];
//   for j = 0..i-1:
//     x[i] -= L[i][j] * x[j];
//   x[i] = x[i] / L[i][i];
//
// Optimized version:
// - still iterates i in forward-substitution order (sequential dependency)
// - accumulates into a local scalar `sum` instead of repeatedly updating x[i]
//   to improve vectorization opportunities
// - uses a sliced traverser over j in [0, i) to avoid a branch `if (j < i)`
//
// All iteration is expressed with Noarr traversers.
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(auto L, auto x, auto b) {
	using namespace noarr;

#pragma scop
	// Outer loop over rows i (forward substitution).
	traverser(L, x, b).template for_dims<'i'>([&](auto inner) {
		// State with current i fixed (only 'i' is present here).
		auto state_i = inner.state();
		const auto i = get_index<'i'>(state_i);

		// Start with RHS value b[i]; we will subtract the dot product into `sum`.
		num_t sum = b[state_i];

		// Create a 1D view of the current row L[i][*].
		// `fix<'i'>(i)` removes the 'i' dimension from L, leaving only 'j'.
		auto L_row = L ^ noarr::fix<'i'>(i);

		// Restrict this row to columns j in [0, i):
		// slice<'j'>(0, i) creates a view with length i in 'j', mapping
		// new j-indices 0..i-1 to original j-indices 0..i-1.
		auto row_trav = noarr::traverser(L_row).order(noarr::slice<'j'>(0, i));

		// Inner loop over j < i:
		//   sum -= L[i][j] * x[j];
		row_trav.for_each([&](auto state_j) {
			const auto j = get_index<'j'>(state_j);
			sum -= L_row[state_j] * x[noarr::idx<'i'>(j)];
		});

		// Diagonal element L[i][i] is in the same row at j == i.
		const num_t diag = L_row[noarr::idx<'j'>(i)];

		// Final update: x[i] = sum / L[i][i];
		x[state_i] = sum / diag;
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// set lengths for dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// allocate data (bags own the memory)
	auto L = noarr::bag(noarr::scalar<num_t>() ^ tuning.l_layout ^ set_lengths);
	auto x = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto b = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize arrays
	init_array(L.get_ref(), x.get_ref(), b.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trisolv(L.get_ref(), x.get_ref(), b.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (prevents dead-code elimination of computation)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x.get_ref());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}