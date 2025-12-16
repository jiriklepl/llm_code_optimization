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
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, int n, auto A, auto B) {
	// A, B: i x j x k
	using namespace noarr;

	// Interior region: i, j, k in [1, n-2]
	auto interior =
		noarr::slice<'i'>(1, n - 2) ^
		noarr::slice<'j'>(1, n - 2) ^
		noarr::slice<'k'>(1, n - 2);

	// Time "dimension" using broadcast: repeats the same spatial domain tsteps times
	auto time_bcast = noarr::bcast<'t'>(tsteps);

	#pragma scop
	traverser(A, B)
		.order(time_bcast)
		.template for_dims<'t'>([=](auto t_trav) {
			// t_trav has 't' fixed; we ignore 't' when indexing A and B
			auto inner = t_trav.order(interior);

			// First sweep: update B from A
			inner.for_each([=](auto state) {
				using noarr::idx;

				// state contains 'i', 'j', 'k', and 't' (ignored by A/B)
				num_t center = A[state];

				num_t lap_i = A[state + idx<'i'>(1)] - num_t(2.0) * center + A[state - idx<'i'>(1)];
				num_t lap_j = A[state + idx<'j'>(1)] - num_t(2.0) * center + A[state - idx<'j'>(1)];
				num_t lap_k = A[state + idx<'k'>(1)] - num_t(2.0) * center + A[state - idx<'k'>(1)];

				B[state] = num_t(0.125) * lap_i
				         + num_t(0.125) * lap_j
				         + num_t(0.125) * lap_k
				         + center;
			});

			// Second sweep: update A from B
			inner.for_each([=](auto state) {
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