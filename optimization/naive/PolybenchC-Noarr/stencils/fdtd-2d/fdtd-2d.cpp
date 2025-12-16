#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/traverser_iter.hpp> // traverser range-based for
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

// -----------------------------------------------------------------------------
// Array initialization
// -----------------------------------------------------------------------------
void init_array(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	// -------------------------------------------------------------------------
	// Initialize fict[t] = t
	// Parallelized over t using a traverser iterator.
	// -------------------------------------------------------------------------
	{
		auto trav_t = traverser(fict);

		#pragma omp parallel for
		for (auto t_trav : trav_t) {
			// t_trav behaves like a state with the current 't' fixed
			auto state = t_trav.state();
			auto t = get_index<'t'>(state);
			fict[state] = (num_t)t;
		}
	}

	// Dimensions (unchanged semantics)
	auto nx = ex | get_length<'i'>();
	auto ny = ex | get_length<'j'>();

	// -------------------------------------------------------------------------
	// Initialize ex, ey, hz on the i x j grid
	// Traversal is i-major, j-minor (rows of contiguous j), and parallelized
	// over the outer i dimension.
	// -------------------------------------------------------------------------
	{
		auto trav = traverser(ex, ey, hz);

		#pragma omp parallel for
		for (auto row_trav : trav) {
			// row_trav has 'i' fixed, iterates over all 'j'
			row_trav.for_each([&](auto state) {
				auto [i, j] = get_indices<'i', 'j'>(state);

				ex[state] = ((num_t)i * (j + 1)) / (num_t)nx;
				ey[state] = ((num_t)i * (j + 2)) / (num_t)ny;
				hz[state] = ((num_t)i * (j + 3)) / (num_t)nx;
			});
		}
	}
}

// -----------------------------------------------------------------------------
// Main computational kernel
// FDTD-2D time stepping using Noarr traversers
// -----------------------------------------------------------------------------
[[gnu::flatten, gnu::noinline]]
void kernel_fdtd_2d(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	auto nx = ex | get_length<'i'>();
	auto ny = ex | get_length<'j'>();

	const num_t half = SCALAR_VAL(0.5);
	const num_t c07  = SCALAR_VAL(0.7);

#pragma scop
	// -------------------------------------------------------------------------
	// Outer time loop over t
	// We keep time stepping strictly sequential to preserve dependencies
	// between time levels, and parallelize the spatial sweeps inside each
	// time step.
	// -------------------------------------------------------------------------
	traverser(fict, ex, ey, hz).template for_dims<'t'>([&](auto trav_t) {
		// State with only the current time index 't' fixed
		auto t_state = trav_t.state();

		// Read fict[t] once per time step to avoid repeated state lookups
		const num_t fict_t = fict[t_state];

		// ---------------------------------------------------------------------
		// 1) Boundary condition:
		//    ey[0][j] = fict[t]  for all j
		// ---------------------------------------------------------------------
		{
			// Fix i = 0, leaving only j as a varying dimension
			auto trav_row0 = trav_t.order(noarr::fix<'i'>(0));

			// Parallelize over j (top-most dimension of trav_row0)
			#pragma omp parallel for
			for (auto j_trav : trav_row0) {
				// j_trav behaves like a state with (t, i = 0, j) fixed
				ey[j_trav] = fict_t;
			}
		}

		// ---------------------------------------------------------------------
		// 2) Update ey in the interior:
		//    ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j])
		//    for i = 1..nx-1, j = 0..ny-1
		//
		// We slice the i-dimension to [1, nx-1] and then iterate i-major,
		// j-minor. Parallelization is over the outer i dimension.
		// ---------------------------------------------------------------------
		{
			auto trav_ey = trav_t.order(noarr::slice<'i'>(1, nx - 1));

			#pragma omp parallel for
			for (auto row_trav : trav_ey) {
				// row_trav has a fixed i in [1, nx-1), iterates over all j
				row_trav.for_each([&](auto state) {
					// state: i in [1, nx-1], j in [0, ny-1]
					auto state_im1j = state - noarr::idx<'i'>(1);

					ey[state] = ey[state]
					            - half * (hz[state] - hz[state_im1j]);
				});
			}
		}

		// ---------------------------------------------------------------------
		// 3) Update ex in the interior:
		//    ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1])
		//    for i = 0..nx-1, j = 1..ny-1
		//
		// We slice the j-dimension to [1, ny-1] and traverse i-major, j-minor.
		// Parallelization is again over i.
		// ---------------------------------------------------------------------
		{
			auto trav_ex = trav_t.order(noarr::slice<'j'>(1, ny - 1));

			#pragma omp parallel for
			for (auto row_trav : trav_ex) {
				// row_trav has a fixed i in [0, nx), iterates over j in [1, ny)
				row_trav.for_each([&](auto state) {
					// state: i in [0, nx-1], j in [1, ny-1]
					auto state_ijm1 = state - noarr::idx<'j'>(1);

					ex[state] = ex[state]
					            - half * (hz[state] - hz[state_ijm1]);
				});
			}
		}

		// ---------------------------------------------------------------------
		// 4) Update hz in the interior:
		//    hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j]
		//                                  + ey[i+1][j] - ey[i][j])
		//    for i = 0..nx-2, j = 0..ny-2
		//
		// We slice both i and j to exclude the last row/column and traverse
		// i-major, j-minor. Parallelization is over i; each thread processes
		// a strip of rows, accessing contiguous j for vectorization.
		// ---------------------------------------------------------------------
		{
			auto trav_hz = trav_t.order(
				noarr::slice<'i'>(0, nx - 1) ^ // i in [0, nx-2]
				noarr::slice<'j'>(0, ny - 1)   // j in [0, ny-2]
			);

			#pragma omp parallel for
			for (auto row_trav : trav_hz) {
				// row_trav has fixed i in [0, nx-2], iterates j in [0, ny-2]
				row_trav.for_each([&](auto state) {
					// state: i in [0, nx-2], j in [0, ny-2]
					auto state_i_jp1 = state + noarr::idx<'j'>(1);
					auto state_ip1_j = state + noarr::idx<'i'>(1);

					hz[state] = hz[state]
					            - c07 * (
						            (ex[state_i_jp1] - ex[state]) +
						            (ey[state_ip1_j] - ey[state])
					            );
				});
			}
		}
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