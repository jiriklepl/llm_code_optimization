#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "atax.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A: i x j
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	// x: j
	DEFINE_PROTO_STRUCT(x_layout, j_vec);
	// y: j
	DEFINE_PROTO_STRUCT(y_layout, j_vec);
	// tmp: i
	DEFINE_PROTO_STRUCT(tmp_layout, i_vec);
} tuning;

// initialization function
void init_array(std::size_t m, std::size_t n, auto A, auto x) {
	using namespace noarr;

	num_t fn = static_cast<num_t>(n);

	// x[j] = 1 + (j / fn)
	traverser(x) | [=](auto sj) {
		auto j = get_index<'j'>(sj);
		x[sj] = num_t(1) + num_t(j) / fn;
	};

	// A[i][j] = ((i + j) % n) / (5*m)
	traverser(A) | [=](auto sij) {
		auto [i, j] = get_indices<'i', 'j'>(sij);
		A[sij] = num_t((i + j) % n) / num_t(5 * m);
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) {
	using namespace noarr;

	#pragma scop
	// y[j] = 0
	traverser(y) | [=](auto sj) {
		y[sj] = num_t(0);
	};

	// for i in M:
	traverser(A, x, y, tmp) | for_dims<'i'>([=](auto inner) {
		// tmp[i] = 0
		tmp[inner.state()] = num_t(0);

		// tmp[i] += A[i][j] * x[j] over j
		inner | for_each<'j'>([=](auto s) {
			tmp[s] += A[s] * x[s];
		});

		// y[j] += A[i][j] * tmp[i] over j
		inner | for_each<'j'>([=](auto s) {
			y[s] += A[s] * tmp[s];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t m = M;
	std::size_t n = N;

	// set lengths proto
	auto set_lengths = noarr::set_length<'i'>(m) ^ noarr::set_length<'j'>(n);

	// bags
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto x   = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y   = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout ^ set_lengths);
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);

	// initialize arrays
	init_array(m, n, A.get_ref(), x.get_ref());

	// run kernel with timing
	auto start = std::chrono::high_resolution_clock::now();
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref());
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (serialize y)
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
