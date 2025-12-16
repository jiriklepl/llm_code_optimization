#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "mvt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// 1D layouts for vectors (all use dimension 'i')
	DEFINE_PROTO_STRUCT(x_layout, i_vec);

	// 2D layout for matrix A: dimensions 'i' (rows) and 'j' (columns)
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	auto n = x1 | get_length<'i'>();

	// Initialize the four vectors
	traverser(x1, x2, y_1, y_2).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x1[state] = (num_t)((i % n)) / n;
		x2[state] = (num_t)(((i + 1) % n)) / n;
		y_1[state] = (num_t)(((i + 3) % n)) / n;
		y_2[state] = (num_t)(((i + 4) % n)) / n;
	});

	// Initialize matrix A: A[i][j] = (i * j % n) / n
	auto nA = A | get_length<'i'>();

	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i * j) % nA) / nA;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_mvt(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	// Views of y_1 and y_2 as vectors over 'j' (needed for y_[*][j])
	auto y_1_j = y_1.get_ref() ^ noarr::rename<'i', 'j'>();
	auto y_2_j = y_2.get_ref() ^ noarr::rename<'i', 'j'>();

	// Transposed view of A to access A[j][i]
	auto A_t = A.get_ref() ^ noarr::rename<'i', 'j', 'j', 'i'>();

#pragma scop
	// First part:
	// for (i = 0; i < N; i++)
	//   for (j = 0; j < N; j++)
	//     x1[i] = x1[i] + A[i][j] * y_1[j];
	noarr::traverser(x1, y_1_j, A).template for_dims<'i'>([=](auto inner) {
		inner.template for_each<'j'>([=](auto state) {
			x1[state] = x1[state] + A[state] * y_1_j[state];
		});
	});

	// Second part:
	// for (i = 0; i < N; i++)
	//   for (j = 0; j < N; j++)
	//     x2[i] = x2[i] + A[j][i] * y_2[j];
	noarr::traverser(x2, y_2_j, A_t).template for_dims<'i'>([=](auto inner) {
		inner.template for_each<'j'>([=](auto state) {
			x2[state] = x2[state] + A_t[state] * y_2_j[state];
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// common length proto-structure for all dimensions
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// data structures
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto x1  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto x2  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);

	// initialize data
	init_array(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_mvt(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (prevents dead-code elimination), if requested
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x1.get_ref());
		noarr::serialize_data(std::cerr, x2.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}