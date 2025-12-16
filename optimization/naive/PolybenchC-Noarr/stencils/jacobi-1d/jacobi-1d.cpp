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

	// length is invariant across the traversal, query it once
	auto n = A | get_length<'i'>();

	traverser(A, B).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		A[state] = (static_cast<num_t>(i) + static_cast<num_t>(2))
		           / static_cast<num_t>(n);
		B[state] = (static_cast<num_t>(i) + static_cast<num_t>(3))
		           / static_cast<num_t>(n);
	});
}

// computation kernel
//
// Optimization:
//  - We keep the time loop and spatial traversal expressed via Noarr traversers.
//  - Within each time step we *fuse* the two spatial sweeps:
//        B[i] = 1/3 * (A[i-1] + A[i] + A[i+1])
//        A[i] = 1/3 * (B[i-1] + B[i] + B[i+1])
//    Instead of doing two complete passes over the interior, we compute B[i]
//    and then immediately (from the second interior point on) compute A[i-1]
//    from a small sliding window of three B-values kept in registers:
//
//      window = (B[i-2], B[i-1], B[i])
//
//    This preserves the exact numerical semantics:
//      A[i] in the original code is 1/3 * (B[i-1] + B[i] + B[i+1]),
//      and in the fused version it is computed at the next iteration using
//      the same three B-values.
//
//  - As a consequence, the second sweep no longer needs to read interior
//    B-elements from memory at all; only the two boundary B values are loaded.
//    This significantly reduces memory traffic while leaving the contents of
//    both A and B identical to the original implementation.
//
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_1d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// spatial problem size
	auto n = A | get_length<'i'>();

	// nothing to do for domains smaller than 3 points:
	// original loops over i in [1, n-2], which is empty for n <= 2
	if (n <= 2)
		return;

	// structure used only to iterate over time steps
	auto time_struct = noarr::scalar<num_t>() ^ noarr::vector<'t'>(tsteps);

	// view selecting interior points i in [1, n-2]
	auto interior = noarr::slice<'i'>(1, n - 2);

	// precompute states for boundary and last interior indices
	auto left_boundary_state  = noarr::idx<'i'>(0);       // i = 0
	auto right_boundary_state = noarr::idx<'i'>(n - 1);   // i = n - 1
	auto last_interior_state  = noarr::idx<'i'>(n - 2);   // i = n - 2

	const num_t coef = static_cast<num_t>(0.33333);

	#pragma scop
	traverser(time_struct).for_each([&](auto /*t_state*/) {
		// Sliding window over B for the second (A-from-B) sweep:
		// After processing the k-th interior point (global i),
		//   B_im2 = B[i-2], B_im1 = B[i-1]
		// and the newly computed B[i] is available as 'newB'.
		num_t B_im2 = num_t{};
		num_t B_im1 = num_t{};
		bool first  = true; // true for the very first interior element (i == 1)

		// Single fused sweep:
		//   - update B from A for all interior points
		//   - for i >= 2, also update A[i-1] from the sliding window of B
		traverser(B, A).order(interior).for_each([&](auto state) {
			// current interior index i is encoded in 'state'
			auto left  = state - noarr::idx<'i'>(1); // i-1
			auto right = state + noarr::idx<'i'>(1); // i+1

			// First sweep: B[i] = 1/3 * (A[i-1] + A[i] + A[i+1])
			const num_t newB = coef * (A[left] + A[state] + A[right]);
			B[state] = newB;

			if (first) {
				// Prime the sliding window for the second sweep.
				// For the very first interior point (i == 1), the required
				// three B-values for A[1] are:
				//   B[0] (left boundary), B[1] (newB), B[2] (next interior)
				// We only know B[0] and B[1] now; B[2] will be known in the
				// next iteration, where we will compute A[1].
				B_im2 = B[left_boundary_state]; // B[0]
				B_im1 = newB;                   // B[1]
				first = false;
			} else {
				// After the first interior element, at this point we have:
				//   B_im2 = B[i-2], B_im1 = B[i-1], newB = B[i]
				// So we can safely compute A[i-1] using the original formula:
				//   A[i-1] = 1/3 * (B[i-2] + B[i-1] + B[i]).
				auto a_state = state - noarr::idx<'i'>(1);
				A[a_state] = coef * (B_im2 + B_im1 + newB);

				// Slide window forward for the next iteration:
				//   next B_im2 = B[i-1], next B_im1 = B[i]
				B_im2 = B_im1;
				B_im1 = newB;
			}
		});

		// Handle the last interior point A[n-2]:
		// After the loop above finishes (and n >= 3), we have:
		//   B_im2 = B[n-3], B_im1 = B[n-2]
		// We still need A[n-2] = 1/3 * (B[n-3] + B[n-2] + B[n-1]),
		// where B[n-1] is the (unchanged) right boundary.
		if (n >= 3) {
			A[last_interior_state] =
				coef * (B_im2 + B_im1 + B[right_boundary_state]);
		}
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