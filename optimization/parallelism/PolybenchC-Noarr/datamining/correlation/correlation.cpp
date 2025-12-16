#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "correlation.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(data_layout, j_vec ^ k_vec);
	DEFINE_PROTO_STRUCT(corr_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) {
	// data: k x j
	using namespace noarr;

	float_n = data | get_length<'k'>();

	traverser(data) | [=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		data[state] = (num_t)(k * j) / (data | get_length<'j'>()) + k;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
	// data: k x j
	// corr: i x j
	// mean: j
	// stddev: j
	using namespace noarr;

	num_t eps = (num_t).1;

	auto corr_ji = corr ^ rename<'i', 'j', 'j', 'i'>();
	auto data_ki = data ^ rename<'j', 'i'>();

	auto ni = corr | get_length<'i'>();

	#pragma scop
	traverser(data, mean) | for_dims<'j'>([=](auto inner) {
		mean[inner] = 0;

		inner | [=](auto state) {
			mean[state] += data[state];
		};

		mean[inner] /= float_n;
	});

	traverser(data, mean, stddev) | for_dims<'j'>([=](auto inner) {
		stddev[inner] = 0;

		inner | [=](auto state) {
			stddev[state] += (data[state] - mean[state]) * (data[state] - mean[state]);
		};

		stddev[inner] /= float_n;
		stddev[inner] = std::sqrt(stddev[inner]);
		stddev[inner] = stddev[inner] <= eps ? (num_t)1.0 : stddev[inner];
	});

	traverser(data, mean, stddev) | [=](auto state) {
		data[state] -= mean[state];
		data[state] /= std::sqrt(float_n) * stddev[state];
	};

	traverser(data, corr, data_ki, corr_ji) ^ span<'i'>(0, ni - 1) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		corr[inner.state() & idx<'j'>(i)] = 1;

		inner ^ shift<'j'>(i + 1) | for_dims<'j'>([=](auto inner) {
			corr[inner] = 0;

			inner | [=](auto state) {
				corr[state] += data_ki[state] * data[state];
			};

			corr_ji[inner] = corr[inner];
		});
	});

	corr[idx<'i'>(ni - 1) & idx<'j'>(ni - 1)] = 1;
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nk = NK;
	std::size_t nj = NJ;

	// data
	num_t float_n;
	auto data = noarr::bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ noarr::set_length<'k'>(nk) ^ noarr::set_length<'j'>(nj));
	auto corr = noarr::bag(noarr::scalar<num_t>() ^ tuning.corr_layout ^ noarr::set_length<'i'>(nj) ^ noarr::set_length<'j'>(nj));
	auto mean = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'j'>(nj));
	auto stddev = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'j'>(nj));

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_correlation(float_n, data.get_ref(), corr.get_ref(), mean.get_ref(), stddev.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, corr.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
