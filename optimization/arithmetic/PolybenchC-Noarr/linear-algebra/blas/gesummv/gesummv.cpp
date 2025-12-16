#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "gesummv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto x) {
	// A: i x j
	// B: i x j
	// x: j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto n = A | get_length<'i'>();

	traverser(A, B, x) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		x[idx<'j'>(i)] = (num_t)(i % n) / n;

		inner | [=](auto state) {
			auto j = get_index<'j'>(state);

			A[state] = (num_t)((i * j + 1) % n) / n;
			B[state] = (num_t)((i * j + 2) % n) / n;
		};
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y) {
	// A: i x j
	// B: i x j
	// tmp: i
	// x: j
	// y: i
	using namespace noarr;

	#pragma scop
	traverser(A, B, tmp, x, y) | for_dims<'i'>([=](auto inner) {
		tmp[inner] = 0;
		y[inner] = 0;

		inner | [=](auto state) {
			tmp[state] += A[state] * x[state];
			y[state] += B[state] * x[state];
		};

		y[inner] = alpha * tmp[inner] + beta * y[inner];
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	num_t alpha;
	num_t beta;

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));
	auto x = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'j'>(n));
	auto y = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), x.get_ref());


	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gesummv(alpha, beta, A.get_ref(), B.get_ref(), tmp.get_ref(), x.get_ref(), y.get_ref());

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
