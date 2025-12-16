#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "covariance.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto m_vec = noarr::vector<'m'>();
constexpr auto n_vec = noarr::vector<'n'>();
constexpr auto p_vec = noarr::vector<'p'>();
constexpr auto q_vec = noarr::vector<'q'>();

struct tuning {
	// data: n x m (N rows, M columns) with row-major layout (m inner, n outer)
	DEFINE_PROTO_STRUCT(data_layout, m_vec ^ n_vec);
	// cov: p x q (M x M) with row-major layout (q inner, p outer)
	DEFINE_PROTO_STRUCT(cov_layout, q_vec ^ p_vec);
	// mean: m (M)
	DEFINE_PROTO_STRUCT(mean_layout, m_vec);
} tuning;

// initialization function
void init_array(num_t &float_n, auto data) {
	using namespace noarr;

	float_n = (num_t)(data | get_length<'n'>());

	traverser(data).for_each([=](auto state) {
		auto i = get_index<'n'>(state);
		auto j = get_index<'m'>(state);
		data[state] = (num_t(i * j)) / (num_t)(data | get_length<'m'>());
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_covariance(int /*m*/, int /*n*/, num_t float_n, auto data, auto cov, auto mean) {
	using namespace noarr;

	#pragma scop
	// mean[j] = sum_i data[i][j] / float_n
	traverser(mean, data).template for_dims<'m'>([=](auto t) {
		auto s_m = t.state();
		mean[s_m] = (num_t)0.0;

		t.for_each([=](auto s) {
			mean[s_m] += data[s]; // s has 'n' and fixed 'm'
		});

		mean[s_m] /= float_n;
	});

	// data[i][j] -= mean[j]
	traverser(data, mean).template for_dims<'n'>([=](auto ti) {
		ti.for_each([=](auto s) {
			data[s] -= mean[s]; // s has 'n' and 'm'
		});
	});

	// cov[i][j] = sum_k data[k][i] * data[k][j] / (float_n - 1); symmetric
	traverser(cov).template for_dims<'p'>([=](auto ti) {
		ti.template for_dims<'q'>([=](auto tij) {
			auto s_ij = tij.state();
			auto i_idx = get_index<'p'>(s_ij);
			auto j_idx = get_index<'q'>(s_ij);

			if (j_idx < i_idx) return; // only upper triangle, fill symmetric later

			cov[s_ij] = (num_t)0.0;

			traverser(data).template for_dims<'n'>([=](auto tk) {
				tk.for_each([=](auto sk) {
					auto k_idx = get_index<'n'>(sk);
					auto v_i = data[noarr::idx<'n', 'm'>(k_idx, i_idx)];
					auto v_j = data[noarr::idx<'n', 'm'>(k_idx, j_idx)];
					cov[s_ij] += v_i * v_j;
				});
			});

			cov[s_ij] /= (float_n - (num_t)1.0);
			cov[noarr::idx<'p', 'q'>(j_idx, i_idx)] = cov[s_ij];
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

	auto set_lengths =
		noarr::set_length<'n'>(n) ^
		noarr::set_length<'m'>(m) ^
		noarr::set_length<'p'>(m) ^
		noarr::set_length<'q'>(m);

	auto data = noarr::bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ set_lengths);
	auto cov  = noarr::bag(noarr::scalar<num_t>() ^ tuning.cov_layout  ^ set_lengths);
	auto mean = noarr::bag(noarr::scalar<num_t>() ^ tuning.mean_layout ^ set_lengths);

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_covariance((int)m, (int)n, float_n, data.get_ref(), cov.get_ref(), mean.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, cov.get_ref() ^ noarr::hoist<'p'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
