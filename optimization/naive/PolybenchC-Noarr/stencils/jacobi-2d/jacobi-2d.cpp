#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "jacobi-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A and B are N x N with C-style row-major layout: i = row, j = column
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;


// Array initialization
void init_array(auto A, auto B) {
	using namespace noarr;

	// A, B: i x j
	traverser(A, B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto n = A | get_length<'i'>();

		A[state] = ((num_t)i * (num_t)(j + 2) + (num_t)2) / (num_t)n;
		B[state] = ((num_t)i * (num_t)(j + 3) + (num_t)3) / (num_t)n;
	});
}


// Main computational kernel
//
// Optimizations vs. original version:
//  - Restrict computation to interior once and then apply tiling (`into_blocks`)
//    in both dimensions through `traverser(...).order(...)`, improving cache
//    locality without changing numerical results.
//  - Reuse a single traverser with a fixed traversal order for all time steps,
//    so the traversal plan is computed once.
//  - Precompute small index-state increments for neighbor access to keep the
//    per-element work minimal while still using Noarr state arithmetic.
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_2d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// A, B: i x j
	std::size_t n = A | get_length<'i'>();

	// No interior points -> nothing to do
	if (tsteps <= 0 || n <= 2)
		return;

	std::size_t inner_len = n - 2; // number of interior points in each dimension

	// Choose a tile size that:
	//  - is at most `inner_len`
	//  - divides `inner_len` (required by into_blocks)
	//  - is a power of two (we keep halving), which tends to work well with caches
	auto choose_block = [inner_len](std::size_t preferred) -> std::size_t {
		std::size_t b = preferred;
		if (b > inner_len)
			b = inner_len;
		if (b == 0)
			b = 1;

		// ensure b divides inner_len; fall back to 1 if needed
		while (b > 1 && (inner_len % b) != 0)
			b >>= 1;

		return b;
	};

	// Use moderate default tiles; they will be reduced if they do not divide inner_len
	const std::size_t block_i = choose_block(32);
	const std::size_t block_j = choose_block(32);

	// Restrict traversal to interior points: i,j in [1, n-2]
	const auto interior =
		noarr::slice<'i'>(1, inner_len) ^
		noarr::slice<'j'>(1, inner_len);

	// Tile both dimensions for better spatial locality.
	// `into_blocks` + `hoist` effectively create a traversal order:
	//   outer: I (blocks in i), J (blocks in j)
	//   inner: i, j within each block
	const auto blocked_order =
		noarr::into_blocks<'i', 'I', 'i'>(block_i) ^ noarr::hoist<'I'>() ^
		noarr::into_blocks<'j', 'J', 'j'>(block_j) ^ noarr::hoist<'J'>();

	// Full traversal order: interior region, then blocked tiling.
	const auto traversal_order = interior ^ blocked_order;

	// Build a traverser once and reuse it for all time steps. The traversal
	// state returned by `for_each` is always expressed in the *original*
	// coordinates (i, j), even though the traversal order is tiled internally.
	auto trav = noarr::traverser(A, B).order(traversal_order);

	// Precompute neighbor index offsets in i and j.
	// These are small, compile-time states that will be added/subtracted
	// from the current index state.
	constexpr auto step_i = noarr::idx<'i'>(1);
	constexpr auto step_j = noarr::idx<'j'>(1);

	constexpr num_t weight = (num_t)0.2;

#pragma scop
	for (int t = 0; t < tsteps; t++) {
		// First sweep: B := stencil(A)
		trav.for_each([&](auto state) {
			// state has (i, j) of an interior point (1 <= i,j <= n-2)
			const auto west  = state - step_j;
			const auto east  = state + step_j;
			const auto south = state + step_i;
			const auto north = state - step_i;

			B[state] = weight * (A[state] + A[west] + A[east] + A[south] + A[north]);
		});

		// Second sweep: A := stencil(B)
		trav.for_each([&](auto state) {
			const auto west  = state - step_j;
			const auto east  = state + step_j;
			const auto south = state + step_i;
			const auto north = state - step_i;

			A[state] = weight * (B[state] + B[west] + B[east] + B[south] + B[north]);
		});
	}
#pragma endscop
}

} // namespace


int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t tsteps = TSTEPS;

	// set lengths of dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// allocate data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_2d((int)tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}