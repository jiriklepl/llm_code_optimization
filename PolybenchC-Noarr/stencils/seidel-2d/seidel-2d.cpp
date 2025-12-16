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
[[gnu::flatten, gnu::noinline]]
void kernel_seidel_2d(int tsteps, auto A) {
	using namespace noarr;

	// grid size in each spatial dimension
	auto n_i = A | get_length<'i'>();
	auto n_j = A | get_length<'j'>();

	#pragma scop
	for (int t = 0; t <= tsteps - 1; t++) {
		// Traverse only the interior points: i, j in [1, n-2]
		// `slice` is applied via `order`, so the state seen in the lambda
		// still uses the original indices (1 .. n-2), suitable for neighbor().
		traverser(A)
			.order(slice<'i'>(1, n_i - 2) ^ slice<'j'>(1, n_j - 2))
			.for_each([&](auto state) {
				auto s = state;

				// Build states for all 8 neighbors and the center
				auto s_im1_jm1 = neighbor<'i', 'j'>(s, -1, -1);
				auto s_im1_j   = neighbor<'i'>(s, -1);
				auto s_im1_jp1 = neighbor<'i', 'j'>(s, -1, +1);

				auto s_i_jm1   = neighbor<'j'>(s, -1);
				auto s_i_j     = s;
				auto s_i_jp1   = neighbor<'j'>(s, +1);

				auto s_ip1_jm1 = neighbor<'i', 'j'>(s, +1, -1);
				auto s_ip1_j   = neighbor<'i'>(s, +1);
				auto s_ip1_jp1 = neighbor<'i', 'j'>(s, +1, +1);

				num_t sum =
					A[s_im1_jm1] + A[s_im1_j] + A[s_im1_jp1] +
					A[s_i_jm1]   + A[s_i_j]   + A[s_i_jp1] +
					A[s_ip1_jm1] + A[s_ip1_j] + A[s_ip1_jp1];

				A[s_i_j] = sum / num_t(9.0);
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