#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "nussinov.hpp"

using num_t = DATA_TYPE;
using base = char;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(table_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(seq_layout, i_vec);
	DEFINE_PROTO_STRUCT(k_layout, k_vec);
} tuning;

inline int match(base b1, base b2) { return ((b1 + b2) == 3 ? 1 : 0); }
inline num_t max_score(num_t s1, num_t s2) { return (s1 >= s2) ? s1 : s2; }

// initialization
void init_array(std::size_t n, auto seq_i, auto table) {
	using namespace noarr;

	traverser(seq_i) | [=](auto s) {
		std::size_t i = get_index<'i'>(s);
		seq_i[s] = (base)((i + 1) % 4);
	};

	traverser(table) | [=](auto s) {
		table[s] = (num_t)0;
	};
}

// printing
void print_array(std::size_t n, auto table) {
	using namespace noarr;

	int t = 0;
	traverser(table) | for_dims<'i'>([&](auto ti) {
		std::size_t i = get_index<'i'>(ti.state());
		ti.order(noarr::shift<'j'>(i)) | [&, i](auto s) {
			if (t % 20 == 0) std::cout << '\n';
			std::cout << table[s] << " ";
			t++;
		};
	});
	std::cout << std::endl;
}

[[gnu::flatten, gnu::noinline]]
void kernel_nussinov(std::size_t n, auto seq_i, auto seq_j, auto table, auto k_struct) {
	using namespace noarr;

	traverser(table, k_struct) | for_dims<'i'>([=](auto ti) {
		// map forward i to original descending i: i_orig = n-1 - i_fwd
		std::size_t i_fwd = get_index<'i'>(ti.state());
		std::size_t i_orig = n - 1 - i_fwd;

		// j from i_orig+1 to n-1 ascending
		ti.order(noarr::slice<'j'>(i_orig + 1, n - (i_orig + 1))) | for_dims<'j'>([=](auto tij) {
			// use state with original i index
			auto s_ij = noarr::update_index<'i'>(tij.state(), [=](auto) { return i_orig; });
			auto j_val = noarr::get_index<'j'>(s_ij);

			// table[i][j] = max(table[i][j], table[i][j-1])
			{
				auto s_ijm1 = noarr::update_index<'j'>(s_ij, [=](auto j) { return (std::size_t)j - 1; });
				table[s_ij] = max_score(table[s_ij], table[s_ijm1]);
			}

			// table[i][j] = max(table[i][j], table[i+1][j])
			if (i_orig + 1 < n) {
				auto s_ip1j = noarr::update_index<'i'>(s_ij, [=](auto) { return i_orig + 1; });
				table[s_ij] = max_score(table[s_ij], table[s_ip1j]);
			}

			// table[i][j] = max(table[i][j], table[i+1][j-1] + match or just table[i+1][j-1])
			if (i_orig + 1 < n) {
				auto s_ip1jm1 = noarr::update_index<'i'>(noarr::update_index<'j'>(s_ij, [=](auto j) { return (std::size_t)j - 1; }), [=](auto) { return i_orig + 1; });
				if (i_orig < j_val - 1) {
					int m = match(seq_i[noarr::idx<'i'>(i_orig)], seq_j[noarr::idx<'j'>(j_val)]);
					table[s_ij] = max_score(table[s_ij], table[s_ip1jm1] + (num_t)m);
				} else {
					table[s_ij] = max_score(table[s_ij], table[s_ip1jm1]);
				}
			}

			// for k = i+1 .. j-1
			std::size_t k_start = i_orig + 1;
			if (j_val > k_start) {
				tij.order(noarr::slice<'k'>(k_start, j_val - k_start)) | [=](auto s_ijk) {
					// state with original i index
					auto s = noarr::update_index<'i'>(s_ijk, [=](auto) { return i_orig; });
					std::size_t k_val = noarr::get_index<'k'>(s);

					auto s_ik = noarr::idx<'i', 'j'>(i_orig, k_val);
					auto s_k1j = noarr::idx<'i', 'j'>(k_val + 1, j_val);

					table[s_ij] = max_score(table[s_ij], table[s_ik] + table[s_k1j]);
				};
			}
		});
	});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	std::size_t n = N;

	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);

	auto table = noarr::bag(noarr::scalar<num_t>() ^ tuning.table_layout ^ set_lengths);
	auto seq_i = noarr::bag(noarr::scalar<base>() ^ tuning.seq_layout ^ set_lengths);
	auto seq_j = noarr::bag(seq_i.get_ref() ^ noarr::rename<'i', 'j'>());

	// dummy k structure for iteration only
	auto k_struct = noarr::scalar<char>() ^ tuning.k_layout ^ set_lengths;

	init_array(n, seq_i.get_ref(), table.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	kernel_nussinov(n, seq_i.get_ref(), seq_j.get_ref(), table.get_ref(), k_struct);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		print_array(n, table.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
