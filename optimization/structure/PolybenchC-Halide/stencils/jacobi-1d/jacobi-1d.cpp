#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
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

	traverser(A) | [=](auto s) {
		auto i = get_index<'i'>(s);
		A[s] = (num_t)(i + 2) / (A | get_length<'i'>());
	};
	traverser(B) | [=](auto s) {
		auto i = get_index<'i'>(s);
		B[s] = (num_t)(i + 3) / (B | get_length<'i'>());
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_1d(int tsteps, auto A, auto B) {
	using namespace noarr;

	const std::size_t n = A | get_length<'i'>();
	if (n < 3) return; // nothing to do

	const auto inner_range = noarr::slice<'i'>(1, n - 2);

	#pragma scop
	for (int t = 0; t < tsteps; t++) {
		traverser(B, A).order(inner_range) | [=](auto s) {
			auto left  = noarr::neighbor<'i'>(s, -1);
			auto right = noarr::neighbor<'i'>(s, +1);
			B[s] = (num_t)0.33333 * (A[left] + A[s] + A[right]);
		};

		traverser(A, B).order(inner_range) | [=](auto s) {
			auto left  = noarr::neighbor<'i'>(s, -1);
			auto right = noarr::neighbor<'i'>(s, +1);
			A[s] = (num_t)0.33333 * (B[left] + B[s] + B[right]);
		};
	}
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	int tsteps = TSTEPS;

	// bags
	auto set_lengths = noarr::set_length<'i'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_1d(tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
