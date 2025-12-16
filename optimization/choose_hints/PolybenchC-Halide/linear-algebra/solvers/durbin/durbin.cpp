#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "durbin.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();

struct tuning {
	DEFINE_PROTO_STRUCT(vec_layout, i_vec);
} tuning;

// initialization function
void init_array(auto r) {
	using namespace noarr;

	auto n = r | get_length<'i'>();
	traverser(r) | [=](auto s) {
		auto i = get_index<'i'>(s);
		r[s] = (num_t)(n + 1 - i);
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_durbin(std::size_t n, auto r, auto y) {
	using namespace noarr;

	// temporary vector z
	auto z = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ noarr::set_length<'i'>(n));

	// scalars
	num_t alpha;
	num_t beta;
	num_t sum;

	#pragma scop
	// initialization
	y[noarr::idx<'i'>(0)] = -r[noarr::idx<'i'>(0)];
	beta = (num_t)1.0;
	alpha = -r[noarr::idx<'i'>(0)];

	// create a synthetic traverser for k in [1, n)
	auto k_struct = noarr::scalar<char>() ^ noarr::vector<'k'>(n);
	noarr::traverser(k_struct).order(noarr::shift<'k'>(1)) | noarr::for_each([&](auto sk) {
		std::size_t k = noarr::get_index<'k'>(sk);

		beta = (num_t)(1.0 - alpha * alpha) * beta;

		sum = (num_t)0.0;
		// sum += r[k-i-1] * y[i] for i in [0, k)
		auto upto_k = noarr::traverser(y).order(noarr::slice<'i'>(0, k));
		upto_k | noarr::for_each([&](auto si) {
			std::size_t i = noarr::get_index<'i'>(si);
			sum += r[noarr::idx<'i'>(k - i - 1)] * y[si];
		});

		alpha = -(r[noarr::idx<'i'>(k)] + sum) / beta;

		// z[i] = y[i] + alpha * y[k-i-1] for i in [0, k)
		upto_k | noarr::for_each([&](auto si) {
			std::size_t i = noarr::get_index<'i'>(si);
			z[si] = y[si] + alpha * y[noarr::idx<'i'>(k - i - 1)];
		});

		// y[i] = z[i] for i in [0, k)
		upto_k | noarr::for_each([&](auto si) {
			y[si] = z[si];
		});

		y[noarr::idx<'i'>(k)] = alpha;
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// input/output data
	auto set_len = noarr::set_length<'i'>(n);
	auto r = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ set_len);
	auto y = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout ^ set_len);

	// initialize data
	init_array(r.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_durbin(n, r.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, y.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
