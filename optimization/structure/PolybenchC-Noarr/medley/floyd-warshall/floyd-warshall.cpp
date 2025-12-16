#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "floyd-warshall.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(path_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto path) {
	using namespace noarr;

	traverser(path).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		num_t v = (num_t)((i * j) % 7 + 1);
		if (((i + j) % 13 == 0) || ((i + j) % 7 == 0) || ((i + j) % 11 == 0))
			v = (num_t)999;

		path[state] = v;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(auto path) {
	using namespace noarr;

	#pragma scop
	auto n = path | get_length<'i'>();

	traverser(path ^ noarr::bcast<'k'>(n)).template for_dims<'k'>([=](auto trav_k) {
		trav_k.template for_dims<'i'>([=](auto trav_i) {
			trav_i.template for_each<'j'>([=](auto state) {
				auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);

				num_t cur = path[state];
				num_t pik = path[noarr::idx<'i', 'j'>(i, k)];
				num_t pkj = path[noarr::idx<'i', 'j'>(k, j)];

				path[state] = (cur < (pik + pkj)) ? cur : (pik + pkj);
			});
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);
	auto path = noarr::bag(noarr::scalar<num_t>() ^ tuning.path_layout ^ set_lengths);

	// initialize
	init_array(path.get_ref());

	// run kernel and time it
	auto start = std::chrono::high_resolution_clock::now();
	kernel_floyd_warshall(path.get_ref());
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, path.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}
