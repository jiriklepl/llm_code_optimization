#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/traverser_iter.hpp> // for range-based traverser iteration

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
//
// Optimization:
//   - Use a traverser iterator to split the work along the outermost
//     spatial dimension 'i'.
//   - Add an OpenMP `parallel for` over that outer loop so that each
//     thread initializes a disjoint i-plane, preserving correctness and
//     improving throughput on multi-core CPUs.
//   - Inside each per-i traverser we still use `for_each` over 'j' and 'k'
//     to stay fully within the Noarr abstraction.
void init_array(int n, auto A, auto B) {
	// A, B: i x j x k, all dimensions of length n
	using namespace noarr;

	auto full_trav = traverser(A, B);

	// Parallelize over the outermost spatial dimension 'i'.
	// Each iteration of this loop gets a traverser with a fixed 'i'
	// and iterates over all (j, k) for that i.
	#pragma omp parallel for schedule(static)
	for (auto i_trav : full_trav) {
		i_trav.for_each([=](auto state) {
			auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);

			// Original initialization:
			// A[i][j][k] = B[i][j][k] =
			//   (DATA_TYPE) (i + j + (n - k)) * 10 / n;
			num_t value = num_t((i + j + (n - k)) * 10) / num_t(n);

			A[state] = value;
			B[state] = value;
		});
	}
}

// computation kernel
//
// Optimizations:
//   - Keep the original 2-sweep Jacobi scheme per time step (B<-A, A<-B)
//     to preserve exact Polybench semantics.
//   - Use a broadcasted "time" dimension 't' so that all data loops are
//     expressed via Noarr traversers.
//   - Restrict computation to the interior region via `slice` proto-structures
//     (i, j, k in [1, n-2]) as before.
//   - Within each time step, introduce an OpenMP-parallel outer loop over
//     the outermost spatial dimension 'i' using traverser iterators.
//     Each thread updates a disjoint i-plane, so there are no write/write
//     conflicts and only read-only sharing of neighbor data.
//   - Inner loops (`for_each`) over (j, k) still see 'k' as the innermost
//     dimension, which keeps accesses contiguous and vectorization-friendly.
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, int n, auto A, auto B) {
	// A, B: i x j x k
	using namespace noarr;

	// Interior region: i, j, k in [1, n-2]
	auto interior =
		noarr::slice<'i'>(1, n - 2) ^
		noarr::slice<'j'>(1, n - 2) ^
		noarr::slice<'k'>(1, n - 2);

	// Time "dimension" using broadcast: repeats the same spatial domain tsteps times.
	// The 't' index is present in the traverser state, but A/B ignore it.
	auto time_bcast = noarr::bcast<'t'>(tsteps);

	#pragma scop
	// First, extend the spatial traversal with the (logical) time dimension.
	auto time_trav = traverser(A, B).order(time_bcast);

	// Outer loop over time steps, expressed via Noarr's for_dims on 't'.
	time_trav.template for_dims<'t'>([=](auto t_trav) {
		// t_trav has 't' fixed to one particular time step; remaining
		// unfixed spatial dimensions are 'i', 'j', 'k'.
		auto inner = t_trav.order(interior);

		// ------------------------------------------------------------------
		// First sweep: update B from A over the interior.
		// ------------------------------------------------------------------
		//
		// We parallelize across the outermost spatial dimension 'i'.
		// The range-based for over `inner` uses the traverser iterator
		// interface, which iterates the topmost dimension (here 'i').
		#pragma omp parallel for schedule(static)
		for (auto i_trav : inner) {
			// i_trav represents a single fixed-`i` slice; inside we traverse
			// all (j, k) for that fixed `i`.
			i_trav.for_each([=](auto state) {
				using noarr::idx;

				// state contains 'i', 'j', 'k', and 't' (ignored by A/B).
				num_t center = A[state];

				num_t lap_i = A[state + idx<'i'>(1)] - num_t(2.0) * center + A[state - idx<'i'>(1)];
				num_t lap_j = A[state + idx<'j'>(1)] - num_t(2.0) * center + A[state - idx<'j'>(1)];
				num_t lap_k = A[state + idx<'k'>(1)] - num_t(2.0) * center + A[state - idx<'k'>(1)];

				B[state] = num_t(0.125) * lap_i
				         + num_t(0.125) * lap_j
				         + num_t(0.125) * lap_k
				         + center;
			});
		}

		// ------------------------------------------------------------------
		// Second sweep: update A from B over the same interior region.
		// ------------------------------------------------------------------
		//
		// Again, parallelize across 'i'; the dependency is only across time
		// (t), so this sweep can safely run in parallel over space.
		#pragma omp parallel for schedule(static)
		for (auto i_trav : inner) {
			i_trav.for_each([=](auto state) {
				using noarr::idx;

				num_t center = B[state];

				num_t lap_i = B[state + idx<'i'>(1)] - num_t(2.0) * center + B[state - idx<'i'>(1)];
				num_t lap_j = B[state + idx<'j'>(1)] - num_t(2.0) * center + B[state - idx<'j'>(1)];
				num_t lap_k = B[state + idx<'k'>(1)] - num_t(2.0) * center + B[state - idx<'k'>(1)];

				A[state] = num_t(0.125) * lap_i
				         + num_t(0.125) * lap_j
				         + num_t(0.125) * lap_k
				         + center;
			});
		}
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