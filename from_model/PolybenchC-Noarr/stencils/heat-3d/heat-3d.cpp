#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "heat-3d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// A and B are 3D: i x j x k, with k the innermost (contiguous) dimension
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(int n, auto A, auto B) {
	// A, B: i x j x k, all dimensions of length n
	using namespace noarr;

	traverser(A, B).for_each([=](auto state) {
		auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);

		// Original: A[i][j][k] = B[i][j][k] =
		//   (DATA_TYPE) (i + j + (n - k)) * 10 / n;
		num_t value = num_t((i + j + (n - k)) * 10) / num_t(n);

		A[state] = value;
		B[state] = value;
	});
}

// computation kernel
//
// Optimizations applied (see models heat-3d-0x):
//  - Keep time stepping sequential.
//  - Restrict computation to interior cells via slices (boundaries fixed).
//  - 3D spatial blocking using noarr::into_blocks_dynamic on i, j, k,
//    with tile indices I, J, K driving the outer loops (better cache reuse).
//  - Preserve k as the innermost, contiguous dimension, so the compiler
//    can efficiently vectorize along k.
//  - Algebraic micro-optimization: factor out the common 0.125 scaling
//    so we compute center + 0.125 * (lap_i + lap_j + lap_k).
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, int n, auto A, auto B) {
	// A, B: i x j x k
	using namespace noarr;

	// --- Spatial interior: i, j, k in [1, n-2] ---
	// We never update the boundary planes (i,j,k ∈ {0, n-1}), they keep
	// their initial values and act as fixed neighbors in the stencil.
	auto interior =
		noarr::slice<'i'>(1, n - 2) ^
		noarr::slice<'j'>(1, n - 2) ^
		noarr::slice<'k'>(1, n - 2);

	// --- 3D spatial blocking for cache locality ---
	// Tile sizes: chosen conservatively; can be tuned per machine.
	// We use into_blocks_dynamic so that non-multiple sizes are handled
	// correctly via an internal guard dimension; no points are skipped.
	constexpr std::size_t tile_i = 16;
	constexpr std::size_t tile_j = 16;
	constexpr std::size_t tile_k = 32;

	auto spatial_blocking =
		noarr::into_blocks_dynamic<'i', 'I', 'i', 'p'>(noarr::lit<tile_i>) ^ // tiles along i
		noarr::into_blocks_dynamic<'j', 'J', 'j', 'q'>(noarr::lit<tile_j>) ^ // tiles along j
		noarr::into_blocks_dynamic<'k', 'K', 'k', 'r'>(noarr::lit<tile_k>);  // tiles along k

	// Note: the additional guard dimensions 'p', 'q', 'r' are used internally
	// by into_blocks_dynamic to mask incomplete tiles. They do not appear in
	// the states we use to index A/B; noarr's traverser only produces states
	// corresponding to valid (i,j,k) points.

	// --- Time "dimension": replicate spatial domain tsteps times ---
	// This keeps the time loop inside the traverser; each logical time step
	// corresponds to one value of 't'. A and B ignore 't' in their layout.
	auto time_bcast = noarr::bcast<'t'>(tsteps);

	// Common stencil constants, factored out to make the intent explicit and
	// to avoid recomputing literal conversions in the inner loops.
	const num_t c   = num_t(0.125); // = 1/8
	const num_t two = num_t(2.0);

	#pragma scop
	noarr::traverser(A, B)
		// The proto-structure passed to order() describes how we traverse:
		//   - time_bcast: adds a sequential 't' dimension
		//   - interior: restricts i,j,k to [1, n-2]
		//   - spatial_blocking: introduces tile indices I,J,K over i,j,k
		.order(time_bcast ^ interior ^ spatial_blocking)
		// Outer loop over time steps (sequential, as required by the scheme)
		.template for_dims<'t'>([=](auto t_trav) {
			// At this point, 't_trav' has 't' fixed. The remaining dimensions
			// include the spatial tile indices I, J, K and, inside each tile,
			// the original spatial indices i, j, k. States passed to A/B still
			// contain only 'i','j','k'; I,J,K are used only for traversal order.

			// --- First sweep: update B from A over all tiles ---
			// This matches the original "inner.for_each" that reads A and writes B.
			t_trav.template for_dims<'I', 'J', 'K'>([=](auto tile_trav) {
				// 'tile_trav' corresponds to one spatial tile for the fixed time t.
				// for_each iterates all (i,j,k) in that tile, with k contiguous.
				tile_trav.for_each([=](auto state) {
					using noarr::idx;

					const num_t center = A[state];

					const num_t lap_i =
						A[state + idx<'i'>(1)] - two * center +
						A[state - idx<'i'>(1)];
					const num_t lap_j =
						A[state + idx<'j'>(1)] - two * center +
						A[state - idx<'j'>(1)];
					const num_t lap_k =
						A[state + idx<'k'>(1)] - two * center +
						A[state - idx<'k'>(1)];

					// Equivalent to original:
					//   B = 0.125*lap_i + 0.125*lap_j + 0.125*lap_k + center
					// but computed as:
					//   B = center + 0.125 * (lap_i + lap_j + lap_k)
					B[state] = center + c * (lap_i + lap_j + lap_k);
				});
			});

			// --- Second sweep: update A from B over all tiles ---
			// This matches the second "inner.for_each" that reads B and writes A.
			// We traverse the same tiles again, creating an implicit barrier
			// between the two sweeps (all B-updates at this t are done before
			// any A-updates at the same t).
			t_trav.template for_dims<'I', 'J', 'K'>([=](auto tile_trav) {
				tile_trav.for_each([=](auto state) {
					using noarr::idx;

					const num_t center = B[state];

					const num_t lap_i =
						B[state + idx<'i'>(1)] - two * center +
						B[state - idx<'i'>(1)];
					const num_t lap_j =
						B[state + idx<'j'>(1)] - two * center +
						B[state - idx<'j'>(1)];
					const num_t lap_k =
						B[state + idx<'k'>(1)] - two * center +
						B[state - idx<'k'>(1)];

					A[state] = center + c * (lap_i + lap_j + lap_k);
				});
			});
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	int n = N;
	int tsteps = TSTEPS;

	// set lengths for spatial dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n) ^
		noarr::set_length<'k'>(n);

	// allocate A and B
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(n, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_heat_3d(tsteps, n, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}