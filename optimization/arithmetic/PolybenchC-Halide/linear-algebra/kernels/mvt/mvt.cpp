#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "mvt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec); // A: i x j (row-major)
	DEFINE_PROTO_STRUCT(x_layout, i_vec);         // x: i
	DEFINE_PROTO_STRUCT(y_layout, j_vec);         // y: j
} tuning;

// initialization function
void init_array(auto x1, auto x2, auto y1, auto y2, auto A) {
	using namespace noarr;

	traverser(x1) | [=](auto s) {
		auto i = get_index<'i'>(s);
		auto n = x1 | get_length<'i'>();
		x1[s] = (num_t)((i % n)) / (num_t)n;
	};

	traverser(x2) | [=](auto s) {
		auto i = get_index<'i'>(s);
		auto n = x2 | get_length<'i'>();
		x2[s] = (num_t)(((i + 1) % n)) / (num_t)n;
	};

	traverser(y1) | [=](auto s) {
		auto j = get_index<'j'>(s);
		auto n = y1 | get_length<'j'>();
		y1[s] = (num_t)(((j + 3) % n)) / (num_t)n;
	};

	traverser(y2) | [=](auto s) {
		auto j = get_index<'j'>(s);
		auto n = y2 | get_length<'j'>();
		y2[s] = (num_t)(((j + 4) % n)) / (num_t)n;
	};

	traverser(A) | [=](auto s) {
		auto [i, j] = get_indices<'i', 'j'>(s);
		auto n = A | get_length<'i'>();
		A[s] = (num_t)((i * j) % n) / (num_t)n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_mvt(auto x1, auto x2, auto y1, auto y2, auto A) {
	using namespace noarr;

	#pragma scop
	// First nest:
	// for i
	//   for j
	//     x1[i] += A[i][j] * y1[j];
	traverser(x1, A, y1) | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto s) {
			x1[s] = x1[s] + A[s] * y1[s];
		});
	});

	// Second nest:
	// for i
	//   for j
	//     x2[i] += A[j][i] * y2[j];
	// Create a renamed view so that passing (i,j) yields A[j][i]
	auto A_T = A ^ noarr::rename<'i', 'j', 'j', 'i'>();
	traverser(x2, A_T, y2) | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto s) {
			x2[s] = x2[s] + A_T[s] * y2[s];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// set lengths for all dimensions
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// allocate bags
	auto A  = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto x1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto x2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout ^ set_lengths);
	auto y2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout ^ set_lengths);

	// initialize data
	init_array(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_mvt(x1.get_ref(), x2.get_ref(), y1.get_ref(), y2.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x1.get_ref());
		noarr::serialize_data(std::cout, x2.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
