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
	// scalar<num_t>() ^ (j_vec ^ i_vec) -> row-major in (i, j): j is inner, i is outer
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

	// initialize L: strictly lower-triangular part including diagonal
	// Original code:
	//   for i in [0, n):
	//     for j in [0, n):
	//       if (j <= i) L[i,j] = ...
	//
	// Here we keep exactly the same set of (i,j) updates but
	// restrict the j-iteration to [0, i] instead of looping
	// over [0, n) and branching on (j <= i).
	traverser(L).template for_dims<'i'>([=](auto inner) {
		// state with current i fixed
		auto state_i = inner.state();
		auto i = get_index<'i'>(state_i);

		// Number of columns to initialize in this row is (i + 1),
		// so j runs over 0..i inclusive.
		std::size_t j_count = i + 1;

		// Restrict the traversal along 'j' to [0, i] via a slice;
		// this removes the inner if(j <= i) branch while preserving
		// the exact initialization domain.
		inner.order(noarr::slice<'j'>(0, j_count)).for_each([=](auto state) {
			auto j = get_index<'j'>(state);

			L[state] = (num_t)(i + n - j + 1) * (num_t)2 / (num_t)n;
		});
	});
}

// Main computational kernel:
// for i = 0..n-1:
//   x[i] = b[i];
//   for j = 0..i-1:
//     x[i] -= L[i][j] * x[j];
//   x[i] = x[i] / L[i][i];
//
// Optimized formulation (same arithmetic, better locality):
//   for i = 0..n-1:
//     tmp = b[i];
//     for j = 0..i-1:
//       tmp -= L[i][j] * x[j];
//     x[i] = tmp / L[i][i];
//
// This keeps i as the outermost (sequential) loop, but
//   - removes the inner "if (j < i)" branch by restricting j to [0, i),
//   - keeps x[i] in a scalar accumulator 'tmp' instead of repeatedly
//     loading/storing x[i] from/to memory inside the inner loop.
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(auto L, auto x, auto b) {
	using namespace noarr;

#pragma scop
	// Traverse with 'i' outermost to preserve the forward-substitution
	// dependency x[i] depending on x[0..i-1].
	traverser(L, x, b).template for_dims<'i'>([&](auto inner) {
		// State with current i fixed (only dimension 'i' present here).
		auto state_i = inner.state();
		auto i = get_index<'i'>(state_i);

		// Local accumulator for x[i], kept in a register.
		// Start from b[i] as in the original algorithm.
		num_t xi = b[state_i];

		// Inner triangular loop: j in [0, i)
		//
		// Original code iterated j over the full [0, n) range with
		// an "if (j < i)" guard. Here we use a per-row slice on the
		// 'j' dimension so that only 0..i-1 are traversed. This removes
		// the branch while preserving the exact set of (i, j) pairs.
		inner.order(noarr::slice<'j'>(0, i)).for_each([&](auto state) {
			// state has both 'i' and 'j'
			auto j = get_index<'j'>(state);

			// L[i][j] is addressed by (i, j) from 'state'.
			// x[j] is addressed by an index state over 'i' only.
			xi -= L[state] * x[noarr::idx<'i'>(j)];
		});

		// Final division by the diagonal element L[i][i],
		// then write x[i] exactly once.
		auto diag_state = noarr::idx<'i', 'j'>(i, i);
		xi /= L[diag_state];
		x[state_i] = xi;
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