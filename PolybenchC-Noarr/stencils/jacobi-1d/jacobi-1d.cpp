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

	traverser(A, B).for_each([=](auto state) {
		auto i = get_index<'i'>(state);
		auto n = A | get_length<'i'>();

		A[state] = (static_cast<num_t>(i) + static_cast<num_t>(2)) / static_cast<num_t>(n);
		B[state] = (static_cast<num_t>(i) + static_cast<num_t>(3)) / static_cast<num_t>(n);
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_1d(int tsteps, auto A, auto B) {
	using namespace noarr;

	// spatial problem size
	auto n = A | get_length<'i'>();

	// structure used only to iterate over time steps
	auto time_struct = noarr::scalar<num_t>() ^ noarr::vector<'t'>(tsteps);

	// view selecting interior points i in [1, n-2]
	auto interior = noarr::slice<'i'>(1, n - 2);

	#pragma scop
	traverser(time_struct).for_each([=](auto /*t_state*/) {
		// first sweep: update B from A
		traverser(B, A).order(interior).for_each([=](auto state) {
			auto left  = state - noarr::idx<'i'>(1);
			auto right = state + noarr::idx<'i'>(1);

			B[state] = static_cast<num_t>(0.33333) *
			           (A[left] + A[state] + A[right]);
		});

		// second sweep: update A from B
		traverser(A, B).order(interior).for_each([=](auto state) {
			auto left  = state - noarr::idx<'i'>(1);
			auto right = state + noarr::idx<'i'>(1);

			A[state] = static_cast<num_t>(0.33333) *
			           (B[left] + B[state] + B[right]);
		});
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