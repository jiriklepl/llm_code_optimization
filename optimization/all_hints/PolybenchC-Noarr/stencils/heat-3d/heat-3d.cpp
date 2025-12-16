#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

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

	// Diffusion coefficient used in the stencil:
	//   new = old + alpha * discrete_laplacian
	static constexpr num_t diffusion_coeff = num_t(0.125);
} tuning;

// initialization function
void init_array(int n, auto A, auto B) {
	// A, B: i x j x k, all dimensions of length n
	using namespace noarr;

	// Precompute 1/n once; per-element we only do a multiplication.
	const num_t inv_n = num_t(1) / num_t(n);

	traverser(A, B).for_each([=](auto state) {
		auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);

		// Original formula:
		//   A[i][j][k] = B[i][j][k] =
		//     (DATA_TYPE) (i + j + (n - k)) * 10 / n;
		//
		// We keep the integer arithmetic identical, but replace the
		// division by a multiplication with the precomputed inv_n.
		int tmp = (i + j + (n - k)) * 10;
		num_t value = num_t(tmp) * inv_n;

		A[state] = value;
		B[state] = value;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, int n, auto A, auto B) {
	// A, B: i x j x k
	using namespace noarr;

	// Interior region: i, j, k in [1, n-2]
	// This excludes the boundary cells, which are kept fixed.
	auto interior =
		noarr::slice<'i'>(1, n - 2) ^
		noarr::slice<'j'>(1, n - 2) ^
		noarr::slice<'k'>(1, n - 2);

	// Time "dimension" using broadcast: repeats the same spatial domain tsteps times.
	// The 't' index is not used for data access, only to drive the time loop.
	auto time_bcast = noarr::bcast<'t'>(tsteps);

	// Precompute scalar coefficients once for the whole kernel.
	const num_t alpha      = tuning.diffusion_coeff; // 0.125
	const num_t minus_six  = num_t(-6.0);            // -6.0 in stencil

	// Precompute unit index states once: +/- 1 step in each spatial dimension.
	const auto di = noarr::idx<'i'>(1);
	const auto dj = noarr::idx<'j'>(1);
	const auto dk = noarr::idx<'k'>(1);

	// The SCoP markers are kept to preserve the original benchmarking setup.
	#pragma scop
	noarr::traverser(A, B)
		// Add broadcasted time dimension.
		.order(time_bcast)
		// Iterate time steps along the synthetic 't' dimension.
		.template for_dims<'t'>([=](auto t_trav) {
			// t_trav has 't' fixed; we ignore 't' when indexing A and B.
			// Restrict traversal to the interior in space.
			auto inner = t_trav.order(interior);

			// -----------------------------------------------------------------
			// First sweep: update B from A
			//
			// Optimization:
			//   - Load the 6 neighbors explicitly and form a single discrete
			//     Laplacian:
			//       lap = sum(neighbors) - 6 * center
			//   - Then compute:
			//       B = center + alpha * lap
			//
			// This is algebraically equivalent to the original
			//   0.125 * lap_i + 0.125 * lap_j + 0.125 * lap_k + center
			// but reduces the number of multiplications and improves
			// data reuse (center is loaded once).
			// -----------------------------------------------------------------
			inner.for_each([=](auto state) {
				const auto s_ip = state + di;
				const auto s_im = state - di;
				const auto s_jp = state + dj;
				const auto s_jm = state - dj;
				const auto s_kp = state + dk;
				const auto s_km = state - dk;

				const num_t center = A[state];

				// Sum of the 6 direct neighbors.
				const num_t neighbor_sum =
					A[s_ip] + A[s_im] +
					A[s_jp] + A[s_jm] +
					A[s_kp] + A[s_km];

				// Discrete Laplacian over i, j, k.
				const num_t lap = neighbor_sum + minus_six * center;

				// New value at (i,j,k) in B.
				B[state] = center + alpha * lap;
			});

			// -----------------------------------------------------------------
			// Second sweep: update A from B
			//
			// Same computation, but reading from B and writing back to A.
			// -----------------------------------------------------------------
			inner.for_each([=](auto state) {
				const auto s_ip = state + di;
				const auto s_im = state - di;
				const auto s_jp = state + dj;
				const auto s_jm = state - dj;
				const auto s_kp = state + dk;
				const auto s_km = state - dk;

				const num_t center = B[state];

				const num_t neighbor_sum =
					B[s_ip] + B[s_im] +
					B[s_jp] + B[s_jm] +
					B[s_kp] + B[s_km];

				const num_t lap = neighbor_sum + minus_six * center;

				A[state] = center + alpha * lap;
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