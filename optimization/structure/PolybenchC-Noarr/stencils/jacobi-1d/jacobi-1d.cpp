#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "jacobi-1d.hpp"

using num_t = DATA_TYPE;

namespace {

// initialization function
void init_array(auto A, auto B) {
	// A: i
	// B: i
	using namespace noarr;

	auto n = A | get_length<'i'>();

	traverser(A, B) | [=](auto state) {
		auto i = get_index<'i'>(state);

		A[state] = ((num_t) i + 2) / n;
		B[state] = ((num_t) i + 3) / n;
	};
}


// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_1d(std::size_t tsteps, auto A, auto B) {
	// A: i
	// B: i
	using namespace noarr;

	auto trav = traverser(A, B) ^ bcast<'t'>(tsteps);

	#pragma scop
	trav | for_dims<'t'>([=](auto inner) {
		inner ^ symmetric_span<'i'>(B, 1) | [=](auto state) {
			B[state] = 0.33333 * (A[state - idx<'i'>(1)] + A[state] + A[state + idx<'i'>(1)]);
		};

		inner ^ symmetric_span<'i'>(A, 1) | [=](auto state) {
			A[state] = 0.33333 * (B[state - idx<'i'>(1)] + B[state] + B[state + idx<'i'>(1)]);
		};
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	// data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));
	auto B = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_1d(t, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
