#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "jacobi-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A and B are N x N with C-style row-major layout: i = row, j = column
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;


// Array initialization
void init_array(auto A, auto B) {
	using namespace noarr;

	// A, B: i x j
	traverser(A, B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto n = A | get_length<'i'>();

		A[state] = ((num_t)i * (num_t)(j + 2) + (num_t)2) / (num_t)n;
		B[state] = ((num_t)i * (num_t)(j + 3) + (num_t)3) / (num_t)n;
	});
}


// Main computational kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_2d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// A, B: i x j
	auto n = A | get_length<'i'>();
	auto inner_len = n - 2; // number of interior points in each dimension

#pragma scop
	for (int t = 0; t < tsteps; t++) {
		// First sweep: B := stencil(A)
		traverser(A)
			.order(noarr::slice<'i'>(1, inner_len) ^ noarr::slice<'j'>(1, inner_len))
			.for_each([=](auto state) {
				auto west  = state - idx<'j'>(1);
				auto east  = state + idx<'j'>(1);
				auto south = state + idx<'i'>(1);
				auto north = state - idx<'i'>(1);

				B[state] = (num_t)0.2 * (A[state] + A[west] + A[east] + A[south] + A[north]);
			});

		// Second sweep: A := stencil(B)
		traverser(B)
			.order(noarr::slice<'i'>(1, inner_len) ^ noarr::slice<'j'>(1, inner_len))
			.for_each([=](auto state) {
				auto west  = state - idx<'j'>(1);
				auto east  = state + idx<'j'>(1);
				auto south = state + idx<'i'>(1);
				auto north = state - idx<'i'>(1);

				A[state] = (num_t)0.2 * (B[state] + B[west] + B[east] + B[south] + B[north]);
			});
	}
#pragma endscop
}

} // namespace


int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t tsteps = TSTEPS;

	// set lengths of dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// allocate data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_2d((int)tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}