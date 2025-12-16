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
// Optimized with 2D spatial blocking using `into_blocks_dynamic` + `hoist`.
//
// We keep the outer time loop sequential (Jacobi dependence) and use Noarr
// traversers to:
//   - restrict to the interior [1, n-1) x [1, n-1) via `slice`,
//   - tile that interior into TI x TJ blocks (`into_blocks_dynamic`),
//   - hoist the tile indices 'I' and 'J' to the top of the iteration order.
//
// Importantly, `order(...)` only changes the traversal order; the `state`
// seen in the lambda still exposes just the original dimensions 'i' and 'j',
// so neighbor indexing (north/south/east/west) is unchanged semantically.
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_2d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// A, B: i x j
	auto n = A | get_length<'i'>();
	auto inner_len = n - 2; // number of interior points in each dimension

	// Tunable tile sizes (rows x columns). Chosen so a tile of A/B plus halos
	// fits comfortably into L1/L2 on typical x64 machines.
	constexpr std::size_t tile_i = 32;
	constexpr std::size_t tile_j = 32;

	// Constant stencil weight (0.2) hoisted out of the inner loops.
	const num_t w = (num_t)0.2;

	// Proto-structure describing:
	//   1. interior restriction: i,j in [1, n-1),
	//   2. blocking of i into (I, i) with guard dim 'r' for partial tiles,
	//   3. blocking of j into (J, j) with guard dim 's' for partial tiles,
	//   4. tiling order: I outermost, then J, then intra-tile i,j.
	//
	// `into_blocks_dynamic` allows N-2 to be non-divisible by the tile sizes.
	// The extra guard dimensions ('r', 's') make the last tiles partially
	// filled; `order(...)` and `for_each` automatically skip the out-of-range
	// points, while the states still expose only 'i' and 'j'.
	auto interior =
		noarr::slice<'i'>(1, inner_len) ^
		noarr::slice<'j'>(1, inner_len);

	auto tiled_order =
		interior
		^ noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(tile_i)
		^ noarr::into_blocks_dynamic<'j', 'J', 'j', 's'>(tile_j)
		^ noarr::hoist<'I'>()
		^ noarr::hoist<'J'>();

#pragma scop
	for (int t = 0; t < tsteps; t++) {
		// First sweep: B := stencil(A) on the interior.
		// The traversal order is:
		//   I (tile rows) -> J (tile cols) -> i (rows inside tile) -> j (cols).
		// Along j we retain unit-stride accesses, which is good for SIMD.
		noarr::traverser(A, B)
			.order(tiled_order)
			.for_each([&](auto state) {
				auto west  = state - idx<'j'>(1);
				auto east  = state + idx<'j'>(1);
				auto south = state + idx<'i'>(1);
				auto north = state - idx<'i'>(1);

				B[state] = w * (A[state] + A[west] + A[east] + A[south] + A[north]);
			});

		// Second sweep: A := stencil(B) on the interior, using the B just computed.
		noarr::traverser(B, A)
			.order(tiled_order)
			.for_each([&](auto state) {
				auto west  = state - idx<'j'>(1);
				auto east  = state + idx<'j'>(1);
				auto south = state + idx<'i'>(1);
				auto north = state - idx<'i'>(1);

				A[state] = w * (B[state] + B[west] + B[east] + B[south] + B[north]);
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