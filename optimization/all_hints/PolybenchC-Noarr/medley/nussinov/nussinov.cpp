#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "nussinov.hpp"

using num_t = DATA_TYPE;
using base_t = char;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(table_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto seq, auto table) {
	// seq: i
	// table: i x j
	using namespace noarr;

	traverser(seq) | [=](auto state) {
		auto i = get_index<'i'>(state);
		seq[state] = (base_t)((i + 1) % 4);
	};

	traverser(table) | [=](auto state) {
		table[state] = 0;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_nussinov(auto seq, auto table) {
	// seq: i
	// table: i x j
	using namespace noarr;

	auto seq_j = seq ^ rename<'i', 'j'>();
	auto table_ik = table ^ rename<'j', 'k'>();
	auto table_kj = table ^ rename<'i', 'k'>();

	#pragma scop
	traverser(seq, table, table_ik, table_kj) ^ reverse<'i'>() | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);
		inner ^ shift<'j'>(i + 1) | for_dims<'j'>([=](auto inner) {
			auto state = inner.state();
			auto [i, j] = get_indices<'i', 'j'>(state);
			auto ni = table | get_length<'i'>();

			if (j >= 0)
				table[state] = max_score(
					table[state],
					table[state - idx<'j'>(1)]);

			if (i + 1 < ni)
				table[state] = max_score(
					table[state],
					table[state + idx<'i'>(1)]);

			if (j >= 0 || i + 1 < ni) {
				if (i < j - 1)
					table[state] = max_score(
						table[state],
						table[state + idx<'i'>(1) - idx<'j'>(1)] +
						match(seq[state], seq_j[state]));
				else
					table[state] = max_score(
						table[state],
						table[state + idx<'i'>(1) - idx<'j'>(1)]);
			}

			inner ^ span<'k'>(i + 1, j) | [=](auto state) {
				table[state] = max_score(
					table[state],
					table_ik[state] +
					table_kj[state + idx<'k'>(1)]);
			};
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
	auto seq = noarr::bag(noarr::scalar<base_t>() ^ noarr::vector<'i'>(n));
	auto table = noarr::bag(noarr::scalar<num_t>() ^ tuning.table_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(seq.get_ref(), table.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_nussinov(seq.get_ref(), table.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) [table = table.get_ref()] {
		std::cout << std::fixed << std::setprecision(2);
		noarr::traverser(table) | noarr::for_dims<'i'>([=](auto inner) {
			std::cout << std::fixed << std::setprecision(2);
			inner ^ noarr::shift<'j'>(noarr::get_index<'i'>(inner)) | noarr::for_each<'j'>([=](auto state) {
				std::cout << table[state] << " ";
			});
		});
	}();

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
