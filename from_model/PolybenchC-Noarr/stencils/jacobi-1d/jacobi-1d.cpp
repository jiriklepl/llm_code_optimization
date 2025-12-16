#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "jacobi-1d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, i_vec);
	DEFINE_PROTO_STRUCT(b_layout, i_vec);
} tuning;

// initialization function
void init_array(auto A, auto B) {
	using namespace noarr;

	// Hoist the length query out of the loop
	auto n = A | get_length<'i'>();

	traverser(A, B).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		// Same arithmetic as the original code; just avoid re-querying the length
		A[state] = (static_cast<num_t>(i) + static_cast<num_t>(2)) /
		           static_cast<num_t>(n);
		B[state] = (static_cast<num_t>(i) + static_cast<num_t>(3)) /
		           static_cast<num_t>(n);
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_1d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// Spatial problem size
	auto n = A | get_length<'i'>();

	// Constant stencil weight (Jacobi averaging coefficient)
	const num_t w = static_cast<num_t>(0.33333);

	// Structure used only to iterate over time steps: t in [0, tsteps)
	auto time_struct = noarr::scalar<num_t>() ^ noarr::vector<'t'>(tsteps);

	// View selecting interior points i in [1, n-2]; boundaries i=0 and i=n-1 stay fixed
	auto interior = noarr::slice<'i'>(1, n - 2);

	// --------------------------------------------------------------------
	// Spatial tiling of the interior along 'i'
	//
	// We block the interior region into tiles of (up to) tile_i elements.
	// This improves cache locality on long domains and exposes a natural
	// outer loop over tiles ('I') that could be parallelized if desired.
	//
	// into_blocks_dynamic<'i','I','i','g'> creates:
	//   - a block index dimension 'I' (tile number)
	//   - a per-tile index 'i'
	//   - a guard dimension 'g' to handle a potentially partial last tile
	//
	// We then hoist 'I' to the top, so the traversal order becomes:
	//   time t  ->  tile I  ->  points i within that tile
	//
	// IMPORTANT: this only changes the traversal order; the logical
	// index state we see in the lambdas still contains the original
	// global 'i' index, so the neighbor accesses (i-1, i+1) remain valid.
	// --------------------------------------------------------------------
	const std::size_t tile_i = 4096; // tunable tile size for cache / prefetch

	auto interior_tiled =
		interior
		^ noarr::into_blocks_dynamic<'i', 'I', 'i', 'g'>(tile_i)
		^ noarr::hoist<'I'>();

	// Pre-build traversers with the chosen spatial order so we don’t
	// reconstruct them for each time step.
	auto trav_B_from_A = traverser(B, A).order(interior_tiled);
	auto trav_A_from_B = traverser(A, B).order(interior_tiled);

	#pragma scop
	traverser(time_struct).for_each([=](auto /*t_state*/) {
		// -----------------------------------------------------------------
		// First sweep: update B from A on interior points
		//   B[i] = w * (A[i-1] + A[i] + A[i+1]), for i = 1 .. n-2
		//
		// Traversal order (conceptually):
		//   for each tile I
		//     for each i in tile I (in increasing global i)
		// -----------------------------------------------------------------
		trav_B_from_A.template for_dims<'I'>([=](auto tile_trav) {
			// 'tile_trav' has the current tile 'I' fixed; only 'i' remains.
			tile_trav.for_each([=](auto state) {
				// 'state' contains the global interior index 'i' ∈ [1, n-2]
				auto left  = state - noarr::idx<'i'>(1);
				auto right = state + noarr::idx<'i'>(1);

				B[state] = w * (A[left] + A[state] + A[right]);
			});
		});

		// -----------------------------------------------------------------
		// Second sweep: update A from B on the same interior
		//   A[i] = w * (B[i-1] + B[i] + B[i+1]), for i = 1 .. n-2
		//
		// We reuse exactly the same spatial tiling and traversal order.
		// The entire first sweep (all i, all tiles) is completed before
		// this second sweep starts, so the temporal dependencies are
		// preserved exactly.
		// -----------------------------------------------------------------
		trav_A_from_B.template for_dims<'I'>([=](auto tile_trav) {
			tile_trav.for_each([=](auto state) {
				auto left  = state - noarr::idx<'i'>(1);
				auto right = state + noarr::idx<'i'>(1);

				A[state] = w * (B[left] + B[state] + B[right]);
			});
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size and number of time steps
	std::size_t n = N;
	std::size_t tsteps = TSTEPS;

	// set the length of the spatial dimension
	auto set_lengths = noarr::set_length<'i'>(n);

	// allocate bags for A and B
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_1d(static_cast<int>(tsteps), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to inhibit dead-code elimination in benchmarks)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}