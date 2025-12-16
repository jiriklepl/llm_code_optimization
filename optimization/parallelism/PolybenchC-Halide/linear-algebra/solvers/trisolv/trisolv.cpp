#include <chrono>
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
	// L: i x j (row-major -> j inner, i outer)
	DEFINE_PROTO_STRUCT(L_layout, j_vec ^ i_vec);
	// x, b: 1D over i
	DEFINE_PROTO_STRUCT(x_layout, i_vec);
	DEFINE_PROTO_STRUCT(b_layout, i_vec);
} tuning;

// initialization function
void init_array(int n, auto L, auto x, auto b) {
	using namespace noarr;

	traverser(x) | [=](auto s) {
		x[s] = (num_t)-999;
	};

	traverser(b) | [=](auto s) {
		auto i = get_index<'i'>(s);
		b[s] = (num_t)i;
	};

	traverser(L) | for_dims<'i'>([=](auto ti) {
		ti | for_each<'j'>([=](auto s) {
			auto i = get_index<'i'>(s);
			auto j = get_index<'j'>(s);
			if (j <= i) {
				L[s] = (num_t(i + n - j + 1) * (num_t)2) / (num_t)n;
			}
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(int /*n*/, auto L, auto x, auto b) {
	using namespace noarr;

	#pragma scop
	traverser(L, x, b) | for_dims<'i'>([=](auto inner) {
		auto si = inner.state(); // contains fixed 'i'

		// x[i] = b[i];
		x[si] = b[si];

		// for (j = 0; j < i; ++j) x[i] -= L[i][j] * x[j];
		auto i_val = get_index<'i'>(si);
		inner.order(noarr::slice<'j'>(0, i_val)) | for_each<'j'>([&](auto s) {
			auto j = get_index<'j'>(s);
			x[si] -= L[s] * x[noarr::idx<'i'>(j)];
		});

		// x[i] = x[i] / L[i][i];
		x[si] = x[si] / L[si + noarr::idx<'j'>(i_val)];
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// set lengths for dimensions
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// allocate bags
	auto L = noarr::bag(noarr::scalar<num_t>() ^ tuning.L_layout ^ set_lengths);
	auto x = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto b = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array((int)n, L.get_ref(), x.get_ref(), b.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trisolv((int)n, L.get_ref(), x.get_ref(), b.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to avoid DCE)
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
