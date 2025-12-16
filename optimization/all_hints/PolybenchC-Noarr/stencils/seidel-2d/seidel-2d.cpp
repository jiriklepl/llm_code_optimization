#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "seidel-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A) {
	// A: i x j
	using namespace noarr;

	auto n = A | get_length<'i'>();

	traverser(A) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		A[state] = ((num_t)i * (j + 2) + 2) / n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_seidel_2d(std::size_t tsteps, auto A) {
	// A: i x j
	using namespace noarr;

	auto trav = traverser(A) ^ bcast<'t'>(tsteps);

	#pragma scop
	trav ^ symmetric_spans<'i', 'j'>(A, 1, 1) ^ reorder<'t', 'i', 'j'>() | [=](auto state) {
		A[state] = (
			A[state - idx<'i'>(1) - idx<'j'>(1)] + // corner
			A[state - idx<'i'>(1)] +          // edge
			A[state - idx<'i'>(1) + idx<'j'>(1)] + // corner
			A[state - idx<'j'>(1)] +          // edge
			A[state] +                             // center
			A[state + idx<'j'>(1)] +          // edge
			A[state + idx<'i'>(1) - idx<'j'>(1)] + // corner
			A[state + idx<'i'>(1)] +          // edge
			A[state + idx<'i'>(1) + idx<'j'>(1)]) / (num_t)9.0; // corner
	};
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	// data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_seidel_2d(t, A.get_ref());

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
