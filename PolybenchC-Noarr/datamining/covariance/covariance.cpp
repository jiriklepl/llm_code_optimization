#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "covariance.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// data: k x j  (N samples, M features)
	DEFINE_PROTO_STRUCT(data_layout, j_vec ^ k_vec);
	// cov:  i x j  (M x M)
	DEFINE_PROTO_STRUCT(cov_layout, j_vec ^ i_vec);
	// mean: j      (M)
	DEFINE_PROTO_STRUCT(mean_layout, j_vec);
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) {
	// data: k x j  (k ~ N, j ~ M)
	using namespace noarr;

	// float_n = N (number of samples)
	float_n = (num_t)(data | get_length<'k'>());

	auto m_len = data | get_length<'j'>();

	traverser(data).template for_dims<'k'>([=](auto tk) {
		tk.template for_each<'j'>([=](auto state) {
			auto [k, j] = get_indices<'k', 'j'>(state);
			data[state] = (num_t)(k * j) / (num_t)m_len;
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_covariance(num_t float_n, auto data, auto cov, auto mean) {
	// data: k x j  (N samples, M features)
	// cov:  i x j  (M x M)
	// mean: j      (M)
	using namespace noarr;

	// view of data where the feature dimension is called 'i'
	// (for cov's row index); data_i: k x i
	auto data_i = data ^ noarr::rename<'j', 'i'>();

#pragma scop
	// 1) mean[j] = (1/n) * sum_k data[k][j]
	traverser(data, mean).template for_dims<'j'>([=](auto tj) {
		// initialize mean[j]
		mean[tj] = (num_t)0;

		// accumulate over k (samples)
		tj.template for_each<'k'>([=](auto state) {
			mean[state] += data[state];
		});

		// divide by float_n
		mean[tj] /= float_n;
	});

	// 2) data[k][j] -= mean[j]
	traverser(data, mean).template for_dims<'k'>([=](auto tk) {
		tk.template for_each<'j'>([=](auto state) {
			data[state] -= mean[state];
		});
	});

	// 3) covariance
	//    for i in 0..M-1
	//      for j in i..M-1
	//        cov[i][j] = sum_k data[k][i] * data[k][j] / (float_n - 1)
	//        cov[j][i] = cov[i][j]
	traverser(cov, data, data_i).template for_dims<'i'>([=](auto ti) {
		ti.template for_dims<'j'>([=](auto tij) {
			auto s_ij = tij.state();
			auto [i_idx, j_idx] = get_indices<'i', 'j'>(s_ij);

			// enforce j >= i (upper triangle, including diagonal)
			if (j_idx < i_idx)
				return;

			// initialize cov[i][j]
			cov[s_ij] = (num_t)0;

			// accumulate over k (samples)
			tij.template for_each<'k'>([=](auto sk) {
				cov[s_ij] += data[sk] * data_i[sk];
			});

			// normalize
			cov[s_ij] /= (float_n - (num_t)1.0);

			// symmetry: cov[j][i] = cov[i][j]
			auto s_ji = noarr::idx<'i', 'j'>(j_idx, i_idx);
			cov[s_ji] = cov[s_ij];
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t m = M;

	// input data
	num_t float_n;

	// set lengths for all relevant dimensions
	auto set_lengths =
		noarr::set_length<'k'>(n) ^ // samples
		noarr::set_length<'i'>(m) ^ // cov rows
		noarr::set_length<'j'>(m);  // features / cov cols

	// allocate bags
	auto data = noarr::bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ set_lengths);
	auto cov  = noarr::bag(noarr::scalar<num_t>() ^ tuning.cov_layout  ^ set_lengths);
	auto mean = noarr::bag(noarr::scalar<num_t>() ^ tuning.mean_layout ^ set_lengths);

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_covariance(float_n, data.get_ref(), cov.get_ref(), mean.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results to stderr (to prevent DCE in benchmarking harnesses)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, cov.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing to stdout
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}