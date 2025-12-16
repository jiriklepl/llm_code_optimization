#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "fdtd-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto t_vec = noarr::vector<'t'>();

struct tuning {
	// 2D fields: i x j, with j the inner (contiguous) dimension
	DEFINE_PROTO_STRUCT(field_layout, j_vec ^ i_vec);
	// 1D fictitious array over time
	DEFINE_PROTO_STRUCT(fict_layout, t_vec);
} tuning;

// Array initialization
void init_array(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	// Initialize fict[t] = t
	traverser(fict).for_each([=](auto state) {
		auto t = get_index<'t'>(state);
		fict[state] = (num_t)t;
	});

	// Dimensions
	auto nx = ex | get_length<'i'>();
	auto ny = ex | get_length<'j'>();

	// Initialize ex, ey, hz
	traverser(ex, ey, hz).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		ex[state] = ((num_t)i * (j + 1)) / (num_t)nx;
		ey[state] = ((num_t)i * (j + 2)) / (num_t)ny;
		hz[state] = ((num_t)i * (j + 3)) / (num_t)nx;
	});
}

// Main computational kernel
//
// Optimizations compared to the original kernel:
//
//  - Fuse the interior parts of ey and ex updates (for i>=1, j>=1) so that
//    hz[i,j] is loaded once and reused in both updates.
//  - Spatially tile the interior (i,j) space for those fused updates and for
//    the hz update using noarr::into_blocks_dynamic + noarr::hoist, improving
//    cache locality while keeping j as the innermost, contiguous dimension.
//  - Hoist fict[t] out of the boundary loop so it is loaded once per t.
//  - Keep all loops expressed via noarr::traverser / order, preserve original
//    FDTD semantics exactly, including boundary handling.
//
[[gnu::flatten, gnu::noinline]]
void kernel_fdtd_2d(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	auto nx = ex | get_length<'i'>();
	auto ny = ex | get_length<'j'>();

	// Tile sizes along i and j (chosen to keep 3 * TI * TJ * sizeof(num_t)
	// comfortably in L1/L2). They are compile-time constants so the compiler
	// can fully optimize the blocking transformation.
	constexpr std::size_t tile_i = 32;
	constexpr std::size_t tile_j = 64;

	// BLAS-like params in original:
	// Time-stepping FDTD on a 2D grid

#pragma scop
	// Outer time loop over t; remains strictly sequential.
	traverser(fict, ex, ey, hz).template for_dims<'t'>([=](auto trav_t) {
		// State with only the current time index
		auto t_state = trav_t.state();

		// Load fict[t] once per time step and reuse
		const num_t ft = fict[t_state];

		// ------------------------------------------------------------------
		// Stage 1: ey boundary at i = 0
		// ey[0][j] = fict[t] for all j
		// ------------------------------------------------------------------
		trav_t
			.order(noarr::fix<'i'>(0))
			.for_each([=](auto state) {
				// state has i = 0, all j, and t (ignored by ey)
				ey[state] = ft;
			});

		// If the grid is degenerate in either dimension, fall back to the
		// original unfused traversal (simpler and avoids corner-case issues
		// with very small domains).
		if (nx <= 1 || ny <= 1) {
			// ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j])
			// for i = 1..nx-1, j = 0..ny-1
			trav_t
				.order(noarr::slice<'i'>(1, nx - 1))
				.for_each([=](auto state) {
					auto state_im1j = state - noarr::idx<'i'>(1);

					ey[state] = ey[state]
					            - SCALAR_VAL(0.5) * (hz[state] - hz[state_im1j]);
				});

			// ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1])
			// for i = 0..nx-1, j = 1..ny-1
			trav_t
				.order(noarr::slice<'j'>(1, ny - 1))
				.for_each([=](auto state) {
					auto state_ijm1 = state - noarr::idx<'j'>(1);

					ex[state] = ex[state]
					            - SCALAR_VAL(0.5) * (hz[state] - hz[state_ijm1]);
				});

			// hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j]
			//                             + ey[i+1][j] - ey[i][j])
			// for i = 0..nx-2, j = 0..ny-2
			trav_t
				.order(noarr::slice<'i'>(0, nx - 1) ^ noarr::slice<'j'>(0, ny - 1))
				.for_each([=](auto state) {
					auto state_i_jp1 = state + noarr::idx<'j'>(1);
					auto state_ip1_j = state + noarr::idx<'i'>(1);

					hz[state] = hz[state]
					            - SCALAR_VAL(0.7) * (
						            (ex[state_i_jp1] - ex[state]) +
						            (ey[state_ip1_j] - ey[state])
					            );
				});

			// Done with this time step
			return;
		}

		// From here on we know nx > 1 and ny > 1.

		// ------------------------------------------------------------------
		// Stage 2: remaining ey boundary column at j = 0
		// ey[i][0] = ey[i][0] - 0.5 * (hz[i][0] - hz[i-1][0])
		// for i = 1..nx-1
		// ------------------------------------------------------------------
		trav_t
			.order(noarr::slice<'i'>(1, nx - 1) ^ noarr::fix<'j'>(0))
			.for_each([=](auto state) {
				// state: i in [1, nx-1], j = 0
				auto state_im1_j = state - noarr::idx<'i'>(1);

				ey[state] = ey[state]
				            - SCALAR_VAL(0.5) * (hz[state] - hz[state_im1_j]);
			});

		// ------------------------------------------------------------------
		// Stage 3: fused interior ey and ex updates for i >= 1, j >= 1
		//
		// ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j])
		// ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1])
		//
		// for i = 1..nx-1, j = 1..ny-1.
		//
		// We tile this interior region in both i and j using
		// into_blocks_dynamic + hoist, which:
		//   - preserves the logical indices 'i' and 'j' in the state,
		//   - traverses tiles in a cache-friendly order with j still
		//     the innermost, contiguous dimension.
		// ------------------------------------------------------------------
		auto interior_order =
			noarr::slice<'i'>(1, nx - 1) ^   // restrict i to [1, nx-1]
			noarr::slice<'j'>(1, ny - 1) ^   // restrict j to [1, ny-1]
			noarr::into_blocks_dynamic<'i', 'I', 'i', 's'>(noarr::lit<tile_i>) ^
			noarr::into_blocks_dynamic<'j', 'J', 'j', 't'>(noarr::lit<tile_j>) ^
			noarr::hoist<'I'>() ^
			noarr::hoist<'J'>();

		trav_t
			.order(interior_order)
			.for_each([=](auto state) {
				// state: i in [1, nx-1], j in [1, ny-1]
				auto state_im1j = state - noarr::idx<'i'>(1);
				auto state_ijm1 = state - noarr::idx<'j'>(1);

				// Load hz[i][j] once and reuse for both updates
				const num_t hz_ij = hz[state];

				ey[state] = ey[state]
				            - SCALAR_VAL(0.5) * (hz_ij - hz[state_im1j]);

				ex[state] = ex[state]
				            - SCALAR_VAL(0.5) * (hz_ij - hz[state_ijm1]);
			});

		// ------------------------------------------------------------------
		// Stage 4: remaining ex boundary row at i = 0, j >= 1
		// ex[0][j] = ex[0][j] - 0.5 * (hz[0][j] - hz[0][j-1])
		// for j = 1..ny-1
		// ------------------------------------------------------------------
		trav_t
			.order(noarr::fix<'i'>(0) ^ noarr::slice<'j'>(1, ny - 1))
			.for_each([=](auto state) {
				// state: i = 0, j in [1, ny-1]
				auto state_ijm1 = state - noarr::idx<'j'>(1);

				ex[state] = ex[state]
				            - SCALAR_VAL(0.5) * (hz[state] - hz[state_ijm1]);
			});

		// ------------------------------------------------------------------
		// Stage 5: hz update using new ex and ey
		//
		// hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j]
		//                              + ey[i+1][j] - ey[i][j])
		// for i = 0..nx-2, j = 0..ny-2.
		//
		// Again we tile in both i and j for cache locality.
		// ------------------------------------------------------------------
		auto hz_order =
			noarr::slice<'i'>(0, nx - 1) ^   // i in [0, nx-2]
			noarr::slice<'j'>(0, ny - 1) ^   // j in [0, ny-2]
			noarr::into_blocks_dynamic<'i', 'I', 'i', 's'>(noarr::lit<tile_i>) ^
			noarr::into_blocks_dynamic<'j', 'J', 'j', 't'>(noarr::lit<tile_j>) ^
			noarr::hoist<'I'>() ^
			noarr::hoist<'J'>();

		trav_t
			.order(hz_order)
			.for_each([=](auto state) {
				// state: i in [0, nx-2], j in [0, ny-2]
				auto state_i_jp1 = state + noarr::idx<'j'>(1);
				auto state_ip1_j = state + noarr::idx<'i'>(1);

				hz[state] = hz[state]
				            - SCALAR_VAL(0.7) * (
					            (ex[state_i_jp1] - ex[state]) +
					            (ey[state_ip1_j] - ey[state])
				            );
			});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// Problem size
	std::size_t tmax = TMAX;
	std::size_t nx = NX;
	std::size_t ny = NY;

	// Set lengths for spatial and temporal dimensions
	auto set_space = noarr::set_length<'i'>(nx) ^ noarr::set_length<'j'>(ny);
	auto set_time  = noarr::set_length<'t'>(tmax);

	// Allocate bags
	auto ex = noarr::bag(noarr::scalar<num_t>() ^ tuning.field_layout ^ set_space);
	auto ey = noarr::bag(noarr::scalar<num_t>() ^ tuning.field_layout ^ set_space);
	auto hz = noarr::bag(noarr::scalar<num_t>() ^ tuning.field_layout ^ set_space);
	auto fict = noarr::bag(noarr::scalar<num_t>() ^ tuning.fict_layout ^ set_time);

	// Initialize arrays
	init_array(ex.get_ref(), ey.get_ref(), hz.get_ref(), fict.get_ref());

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();

	// Run kernel
	kernel_fdtd_2d(ex.get_ref(), ey.get_ref(), hz.get_ref(), fict.get_ref());

	// Stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// Print results (if requested, mimicking PolyBench's conditional printing)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);

		// Hoist 'i' so each row is contiguous in the serialized output
		noarr::serialize_data(std::cerr, ex.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cerr, ey.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cerr, hz.get_ref() ^ noarr::hoist<'i'>());
	}

	// Print elapsed time (seconds)
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}