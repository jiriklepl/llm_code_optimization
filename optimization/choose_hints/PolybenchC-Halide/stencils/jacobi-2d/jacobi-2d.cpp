#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "jacobi-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(int n, auto A, auto B) {
	using namespace noarr;

	traverser(A) | [=](auto s) {
		auto [i, j] = get_indices<'i', 'j'>(s);
		A[s] = ((num_t)i * (j + 2) + (num_t)2) / (num_t)n;
	};

	traverser(B) | [=](auto s) {
		auto [i, j] = get_indices<'i', 'j'>(s);
		B[s] = ((num_t)i * (j + 3) + (num_t)3) / (num_t)n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_jacobi_2d(int tsteps, auto A, auto B) {
	using namespace noarr;

	const num_t c02 = (num_t)0.2;

	const std::size_t ni = A | get_length<'i'>();
	const std::size_t nj = A | get_length<'j'>();

	auto interior = noarr::slice<'i'>(1, ni - 2) ^ noarr::slice<'j'>(1, nj - 2);

	#pragma scop
	for (int t = 0; t < tsteps; t++) {
		traverser(B, A).order(interior) | [=](auto s) {
			auto [i, j] = get_indices<'i', 'j'>(s);
			B[s] = c02 * (
				A[s] +
				A[noarr::idx<'i','j'>(i, j - 1)] +
				A[noarr::idx<'i','j'>(i, j + 1)] +
				A[noarr::idx<'i','j'>(i + 1, j)] +
				A[noarr::idx<'i','j'>(i - 1, j)]
			);
		};

		traverser(A, B).order(interior) | [=](auto s) {
			auto [i, j] = get_indices<'i', 'j'>(s);
			A[s] = c02 * (
				B[s] +
				B[noarr::idx<'i','j'>(i, j - 1)] +
				B[noarr::idx<'i','j'>(i, j + 1)] +
				B[noarr::idx<'i','j'>(i + 1, j)] +
				B[noarr::idx<'i','j'>(i - 1, j)]
			);
		};
	}
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	int n = N;
	int tsteps = TSTEPS;

	// bags
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);

	// initialize data
	init_array(n, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_jacobi_2d(tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (A)
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
