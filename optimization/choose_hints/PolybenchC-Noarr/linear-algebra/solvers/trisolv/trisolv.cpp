#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "trisolv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(l_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto L, auto x, auto b) {
	// L: i x j
	// x: i
	// b: i
	using namespace noarr;

	auto n = L | get_length<'i'>();

	traverser(L, x, b) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		x[inner] = -999;
		b[inner] = i;

		inner ^ span<'j'>(i + 1) | for_each<'j'>([=](auto state) {
			auto j = get_index<'j'>(state);
			L[state] = (num_t)(i + n - j + 1) * 2 / n;
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(auto L, auto x, auto b) {
	// L: i x j
	// x: i
	// b: i
	using namespace noarr;

	auto x_j = x ^ rename<'i', 'j'>();

	#pragma scop
	traverser(L, x, b) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		x[inner] = b[inner];

		inner ^ span<'j'>(i) | [=](auto state) {
			x[state] -= L[state] * x_j[state];
		};

		x[inner] = x[inner] / L[inner.state() & idx<'j'>(i)];
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto L = noarr::bag(noarr::scalar<num_t>() ^ tuning.l_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto x = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));
	auto b = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));

	// initialize data
	init_array(L.get_ref(), x.get_ref(), b.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trisolv(L.get_ref(), x.get_ref(), b.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
