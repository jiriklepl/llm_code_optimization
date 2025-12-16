#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions (DATA_TYPE, N, TSTEPS, DEFINE_PROTO_STRUCT, ...)
#include "defines.hpp"

// include benchmark-specific definitions
#include "seidel-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A is an N x N matrix indexed by (i, j)
	// layout: i = row index (outer), j = column index (inner)
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;


// Array initialization
// A: i x j
void init_array(auto A) {
	using namespace noarr;

	// grid size inferred from the structure
	auto n = A | get_length<'i'>();
	auto n_val = static_cast<num_t>(n);

	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		A[state] = (static_cast<num_t>(i) * (static_cast<num_t>(j) + num_t(2))
		            + num_t(2))
		           / n_val;
	});
}


// Main computational kernel
// A: i x j
//
// Optimizations applied:
//  - Replace division by 9.0 with multiplication by a precomputed 1/9.
//  - 2D spatial blocking (tiling) in i and j via into_blocks_dynamic,
//    expressed through a traverser `.order(...)`. This improves cache
//    locality while preserving the original lexicographic (i,j) order
//    required by the in-place Gauss–Seidel scheme.
//
// The traversal order is:
//
//   for t
//     for tile I in i
//       for i within tile I (increasing)
//         for tile J in j
//           for j within tile J (increasing)
//
// Flattened, this visits all (i,j) in exactly the same lexicographic
// order as the original loops (i outer, j inner), so Gauss–Seidel
// dependencies are preserved.
[[gnu::flatten, gnu::noinline]]
void kernel_seidel_2d(int tsteps, auto A) {
	using namespace noarr;

	// grid size in each spatial dimension
	auto n_i = A | get_length<'i'>();
	auto n_j = A | get_length<'j'>();

	// Precompute reciprocal to avoid a division per point
	const num_t inv9 = num_t(1.0) / num_t(9.0);

	// Tile sizes: chosen to get a few tiles into L1/L2.
	// These are compile-time constants but are passed to into_blocks_dynamic
	// as dynamic values, so they need not divide N.
	constexpr std::size_t tile_i = 32;
	constexpr std::size_t tile_j = 32;

	// Build a proto-structure describing:
	//  - dynamic blocking in 'i' and 'j' (handles edge tiles automatically),
	//  - restriction to the interior [1, n-2] in both dimensions.
	//
	// into_blocks_dynamic<'i','I','i','r'>(tile_i):
	//   - splits dimension 'i' into block index 'I' and intra-block 'i'
	//   - adds guard dimension 'r' that is empty for out-of-range blocks
	//
	// into_blocks_dynamic<'j','J','j','s'>(tile_j):
	//   - analogous for dimension 'j'
	//
	// slice<'i'>(1, n_i-2), slice<'j'>(1, n_j-2):
	//   - restrict to interior points i,j in [1, n-2]
	auto interior_order =
		noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(tile_i) ^
		noarr::into_blocks_dynamic<'j', 'J', 'j', 's'>(tile_j) ^
		noarr::slice<'i'>(std::size_t(1), n_i - 2) ^
		noarr::slice<'j'>(std::size_t(1), n_j - 2);

	// Create a traverser with the above blocking and slicing.
	// NOTE: The state passed to the lambda still uses only the original
	//       dimensions 'i' and 'j'; the tiling dimensions ('I','J','r','s')
	//       are internal to the traverser and only affect traversal order.
	auto trav = traverser(A).order(interior_order);

	#pragma scop
	for (int t = 0; t <= tsteps - 1; t++) {
		// Traverse only the interior points: i, j in [1, n-2],
		// in a cache-friendly tiled order, but with the same
		// global lexicographic (i,j) order as the original code.
		trav.for_each([&](auto state) {
			auto s = state; // current (i,j) index state

			// Build states for all 8 neighbors and the center.
			// We first move in i (up/down), then in j (left/right),
			// composing neighbors to avoid recomputing indices more
			// than needed. All offsets are relative to the same
			// interior point s, as in the original kernel.

			// vertical neighbors at same j
			auto s_im1 = neighbor<'i'>(s, -1); // (i-1, j)
			auto s_ip1 = neighbor<'i'>(s, +1); // (i+1, j)

			// horizontal neighbors at same i
			auto s_jm1 = neighbor<'j'>(s, -1); // (i, j-1)
			auto s_jp1 = neighbor<'j'>(s, +1); // (i, j+1)

			// diagonal neighbors
			auto s_im1_jm1 = neighbor<'j'>(s_im1, -1); // (i-1, j-1)
			auto s_im1_jp1 = neighbor<'j'>(s_im1, +1); // (i-1, j+1)
			auto s_ip1_jm1 = neighbor<'j'>(s_ip1, -1); // (i+1, j-1)
			auto s_ip1_jp1 = neighbor<'j'>(s_ip1, +1); // (i+1, j+1)

			// Load the 9-point neighborhood
			const num_t c_im1_jm1 = A[s_im1_jm1];
			const num_t c_im1_j   = A[s_im1];
			const num_t c_im1_jp1 = A[s_im1_jp1];

			const num_t c_i_jm1   = A[s_jm1];
			const num_t c_i_j     = A[s];
			const num_t c_i_jp1   = A[s_jp1];

			const num_t c_ip1_jm1 = A[s_ip1_jm1];
			const num_t c_ip1_j   = A[s_ip1];
			const num_t c_ip1_jp1 = A[s_ip1_jp1];

			const num_t sum =
				c_im1_jm1 + c_im1_j   + c_im1_jp1 +
				c_i_jm1   + c_i_j     + c_i_jp1 +
				c_ip1_jm1 + c_ip1_j   + c_ip1_jp1;

			// Gauss–Seidel in-place update:
			// new A[i,j] immediately visible to later points in this time step.
			A[s] = sum * inv9;
		});
	}
	#pragma endscop
}

} // namespace


int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	int tsteps = TSTEPS;

	// configure layout and lengths
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_seidel_2d(tsteps, A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (optional, to prevent dead-code elimination / check correctness)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}