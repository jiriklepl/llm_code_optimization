#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "atax.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto x) {
	// A: i x j
	// x: j
	using namespace noarr;

	auto ni = A | get_length<'i'>();
	auto nj = A | get_length<'j'>();

	traverser(x) | [=](auto state) {
		auto j = get_index<'j'>(state);
		x[state] = 1 + j / (num_t)nj;
	};

	traverser(A) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i + j) % nj) / (5 * ni);
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) {
	// A: i x j
	// x: j
	// y: j
	// tmp: i
	using namespace noarr;

	#pragma scop
	traverser(y) | [=](auto state) {
		y[state] = 0;
	};

	traverser(tmp, A, x, y) | for_dims<'i'>([=](auto inner) {
		tmp[inner] = 0;

		inner | [=](auto state) {
			tmp[state] += A[state] * x[state];
		};

		inner | [=](auto state) {
			y[state] += A[state] * tmp[state];
		};
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));

	auto x = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'j'>(nj));
	auto y = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'j'>(nj));

	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(ni));

	// initialize data
	init_array(A.get_ref(), x.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
