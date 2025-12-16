#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "correlation.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto n_vec = noarr::vector<'n'>(); // rows (N)
constexpr auto m_vec = noarr::vector<'m'>(); // cols (M)
constexpr auto p_vec = noarr::vector<'p'>(); // corr row (M)
constexpr auto q_vec = noarr::vector<'q'>(); // corr col (M)

struct tuning {
	DEFINE_PROTO_STRUCT(data_layout, m_vec ^ n_vec);    // data: n x m
	DEFINE_PROTO_STRUCT(corr_layout, q_vec ^ p_vec);    // corr: p x q (both size m)
	DEFINE_PROTO_STRUCT(mean_layout, m_vec);            // mean: m
	DEFINE_PROTO_STRUCT(stddev_layout, m_vec);          // stddev: m
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) {
	using namespace noarr;

	float_n = (num_t)(data | get_length<'n'>());
	auto m_len = data | get_length<'m'>();

	traverser(data) | for_dims<'n'>([=](auto tn) {
		tn | for_each<'m'>([=](auto s) {
			auto i = get_index<'n'>(s);
			auto j = get_index<'m'>(s);
			data[s] = (num_t)((i * j)) / (num_t)m_len + (num_t)i;
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_correlation(int m, int n, num_t float_n, auto data, auto corr, auto mean, auto stddev) {
	using namespace noarr;

	num_t eps = (num_t)0.1;

	#pragma scop
	// mean[j] = sum_i data[i][j] / float_n
	traverser(mean, data) | for_dims<'m'>([=](auto tj) {
		auto s_j = tj.state();
		mean[s_j] = (num_t)0;
		tj | for_dims<'n'>([=](auto ti) {
			mean[s_j] += data[ti.state()];
		});
		mean[s_j] /= float_n;
	});

	// stddev[j] = sqrt( sum_i (data[i][j] - mean[j])^2 / float_n ), clamp to >= 1.0 if <= eps
	traverser(stddev, mean, data) | for_dims<'m'>([=](auto tj) {
		auto s_j = tj.state();
		stddev[s_j] = (num_t)0;
		tj | for_dims<'n'>([=](auto ti) {
			auto s = ti.state();
			num_t diff = data[s] - mean[s_j];
			stddev[s_j] += diff * diff;
		});
		stddev[s_j] /= float_n;
		stddev[s_j] = std::sqrt(stddev[s_j]);
		stddev[s_j] = (stddev[s_j] <= eps) ? (num_t)1.0 : stddev[s_j];
	});

	// Center and reduce: data[i][j] = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
	num_t inv_denom_scale = (num_t)1.0 / std::sqrt(float_n); // we'll multiply by inv_denom_scale/stddev[j]
	traverser(data, mean, stddev) | for_dims<'n'>([=](auto tn) {
		tn | for_each<'m'>([=](auto s) {
			data[s] -= mean[s];
			data[s] *= inv_denom_scale / stddev[s];
		});
	});

	// Correlation matrix
	// for i in [0, m-2]:
	//   corr[i][i] = 1
	//   for j in [i+1, m-1]:
	//     corr[i][j] = sum_k data[k][i] * data[k][j]
	//     corr[j][i] = corr[i][j]
	auto m_len = corr | get_length<'p'>();
	traverser(corr).order(noarr::slice<'p'>(0, m_len > 0 ? (m_len - 1) : 0)) | for_dims<'p'>([=](auto ti) {
		auto s_i = ti.state();
		std::size_t i_idx = get_index<'p'>(s_i);

		// diagonal
		corr[s_i & noarr::idx<'q'>(i_idx)] = (num_t)1.0;

		// j from i+1 to m-1
		auto j_len = corr | get_length<'q'>();
		if (i_idx + 1 <= j_len) {
			ti.order(noarr::slice<'q'>(i_idx + 1, j_len - (i_idx + 1))) | for_each([=](auto s_ij) {
				// accumulate dot product over n
				corr[s_ij] = (num_t)0;
				traverser(data) | for_each<'n'>([=](auto s_n) {
					auto k_state = s_n;
					auto j_idx = get_index<'q'>(s_ij);
					auto s_ki = k_state & noarr::idx<'m'>(i_idx);
					auto s_kj = k_state & noarr::idx<'m'>(j_idx);
					corr[s_ij] += data[s_ki] * data[s_kj];
				});
				// symmetry
				auto j_idx = get_index<'q'>(s_ij);
				corr[noarr::idx<'p','q'>(j_idx, i_idx)] = corr[s_ij];
			});
		}
	});
	// last diagonal element
	if (m_len > 0) {
		auto last = noarr::idx<'p','q'>(m_len - 1, m_len - 1);
		corr[last] = (num_t)1.0;
	}
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

	auto set_lengths =
		noarr::set_length<'n'>(n) ^
		noarr::set_length<'m'>(m) ^
		noarr::set_length<'p'>(m) ^
		noarr::set_length<'q'>(m);

	auto data = noarr::bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ set_lengths);
	auto corr = noarr::bag(noarr::scalar<num_t>() ^ tuning.corr_layout ^ set_lengths);
	auto mean = noarr::bag(noarr::scalar<num_t>() ^ tuning.mean_layout ^ set_lengths);
	auto stddev = noarr::bag(noarr::scalar<num_t>() ^ tuning.stddev_layout ^ set_lengths);

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_correlation((int)m, (int)n, float_n, data.get_ref(), corr.get_ref(), mean.get_ref(), stddev.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, corr.get_ref() ^ noarr::hoist<'p'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
