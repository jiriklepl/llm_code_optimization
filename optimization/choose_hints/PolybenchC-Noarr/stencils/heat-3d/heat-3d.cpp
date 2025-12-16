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

		// A[i][j][k] = B[i][j][k] =
		//   (DATA_TYPE) (i + j + (n - k)) * 10 / n;
		num_t value = num_t((i + j + (n - k)) * 10) / num_t(n);

		A[state] = value;
		B[state] = value;
	});
}

// computation kernel
//
// Optimizations over the original version:
//
// 1. Time is still modeled as a broadcast dimension 't' so that the outer
//    time loop is expressed via a traverser (for_dims<'t'>).
//
// 2. Spatial traversal is restructured to iterate explicitly over (i, j)
//    with for_dims<'i','j'>, leaving k as the only remaining dimension
//    in the inner traverser. This makes it explicit that we are sweeping
//    contiguous k-lines for each (i, j) pair.
//
// 3. Along the innermost (contiguous) k-dimension we use a small
//    "sliding window" for the k-neighbors:
//      prev_k = A[i][j][k-1]
//      center = A[i][j][k]
//      next_k = A[i][j][k+1]
//
//    Instead of re-loading all three values from memory for every k,
//    we reuse prev_k and center across iterations and only load the
//    new next_k. This reduces the number of loads along k from 3 per
//    point to ~1 per point, while keeping the loads in i/j directions
//    unchanged. The same trick is applied in the second sweep.
//
// 4. The arithmetic for the update is slightly rearranged to compute
//       center + alpha * (lap_i + lap_j + lap_k)
//    instead of three separate multiplications by 0.125. This is
//    algebraically equivalent and usually compiled to fewer operations.
//
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, int n, auto A, auto B) {
	using namespace noarr;

	// Interior region: i, j, k in [1, n-2]
	auto interior =
		noarr::slice<'i'>(1, n - 2) ^
		noarr::slice<'j'>(1, n - 2) ^
		noarr::slice<'k'>(1, n - 2);

	// Time "dimension" using broadcast: repeats the same spatial domain tsteps times
	auto time_bcast = noarr::bcast<'t'>(tsteps);

	// Stencil constants, captured into lambdas
	constexpr num_t two   = num_t(2.0);
	constexpr num_t alpha = num_t(0.125);

	#pragma scop
	traverser(A, B)
		.order(time_bcast)
		// Traverse time steps via a logical 't' dimension
		.template for_dims<'t'>([&](auto t_trav) {
			// Apply interior restriction for the current time step
			auto interior_trav = t_trav.order(interior);

			// -----------------------------------------------------------------
			// First sweep: update B from A
			// -----------------------------------------------------------------
			//
			// We iterate over all (i, j) pairs explicitly, and for each of
			// them we sweep the contiguous k-line using a sliding window
			// along k to reduce memory traffic in the k-direction.
			//
			interior_trav.template for_dims<'i', 'j'>([&](auto ij_trav) {
				// ij_trav has 't', 'i', 'j' fixed; it only iterates 'k'
				ij_trav.for_each(
					[&, first = true,
					    prev_k = num_t(0),
					    center_k = num_t(0),
					    next_k = num_t(0)](auto state) mutable {
						using noarr::idx;

						// state contains 't', 'i', 'j', 'k'
						// Sliding window initialization and update:
						//
						//  * On the first k we load (k-1, k, k+1).
						//  * On subsequent k's we shift the window:
						//      prev_k   <- center_k
						//      center_k <- next_k
						//      next_k   <- A[i][j][k+1]
						//
						if (first) {
							prev_k   = A[state - idx<'k'>(1)];
							center_k = A[state];
							next_k   = A[state + idx<'k'>(1)];
							first = false;
						} else {
							prev_k   = center_k;
							center_k = next_k;
							next_k   = A[state + idx<'k'>(1)];
						}

						const num_t center = center_k;

						// Laplacian contributions in i, j, k directions
						const num_t lap_i =
							A[state + idx<'i'>(1)] - two * center + A[state - idx<'i'>(1)];
						const num_t lap_j =
							A[state + idx<'j'>(1)] - two * center + A[state - idx<'j'>(1)];
						const num_t lap_k =
							next_k - two * center + prev_k;

						B[state] = center + alpha * (lap_i + lap_j + lap_k);
					});
			});

			// -----------------------------------------------------------------
			// Second sweep: update A from B
			// -----------------------------------------------------------------
			//
			// Identical traversal pattern, but now B is the input field
			// and A is the output field. We reuse the same sliding-window
			// scheme along k to reduce memory traffic in the k-direction.
			//
			interior_trav.template for_dims<'i', 'j'>([&](auto ij_trav) {
				ij_trav.for_each(
					[&, first = true,
					    prev_k = num_t(0),
					    center_k = num_t(0),
					    next_k = num_t(0)](auto state) mutable {
						using noarr::idx;

						if (first) {
							prev_k   = B[state - idx<'k'>(1)];
							center_k = B[state];
							next_k   = B[state + idx<'k'>(1)];
							first = false;
						} else {
							prev_k   = center_k;
							center_k = next_k;
							next_k   = B[state + idx<'k'>(1)];
						}

						const num_t center = center_k;

						const num_t lap_i =
							B[state + idx<'i'>(1)] - two * center + B[state - idx<'i'>(1)];
						const num_t lap_j =
							B[state + idx<'j'>(1)] - two * center + B[state - idx<'j'>(1)];
						const num_t lap_k =
							next_k - two * center + prev_k;

						A[state] = center + alpha * (lap_i + lap_j + lap_k);
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