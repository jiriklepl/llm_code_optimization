#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "correlation.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto n_vec = noarr::vector<'n'>();
constexpr auto m_vec = noarr::vector<'m'>();
constexpr auto p_vec = noarr::vector<'p'>();
constexpr auto q_vec = noarr::vector<'q'>();

struct tuning {
	// data: n x m  (rows: n, columns/features: m), row-major (m contiguous)
	DEFINE_PROTO_STRUCT(data_layout, m_vec ^ n_vec);
	// corr: p x q (both run over features 0..M-1), row-major (q contiguous)
	DEFINE_PROTO_STRUCT(corr_layout, q_vec ^ p_vec);
	// 1D vectors over features
	DEFINE_PROTO_STRUCT(vec_layout, m_vec);
} tuning;


// initialization function
// float_n <- N (number of rows in data)
// data[i][j] = (DATA_TYPE)(i*j)/M + i;
void init_array(num_t &float_n, auto data) {
	using namespace noarr;

	// N is the length in the 'n' dimension
	float_n = static_cast<num_t>(data | get_length<'n'>());

	traverser(data).for_each([=](auto state) {
		auto [i, j] = get_indices<'n', 'm'>(state);

		// M is the length in the 'm' dimension
		num_t M_len = static_cast<num_t>(data | get_length<'m'>());

		data[state] =
			static_cast<num_t>(static_cast<num_t>(i * j) / M_len + static_cast<num_t>(i));
	});
}


// main computational kernel
[[gnu::flatten, gnu::noinline]]
void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
	using namespace noarr;

	num_t eps = SCALAR_VAL(0.1);

#pragma scop

	// ------------------------------------------------------------------------
	// mean[j] = 1/N * sum_i data[i][j]
	// ------------------------------------------------------------------------
	traverser(data, mean).template for_dims<'m'>([=](auto trav_m) {
		// trav_m has 'm' fixed (one feature column)
		auto state_m = trav_m.state();

		mean[state_m] = SCALAR_VAL(0.0);

		trav_m.for_each([=](auto state) {
			// state has 'n' and 'm'; mean ignores 'n'
			mean[state] += data[state];
		});

		mean[state_m] /= float_n;
	});

	// ------------------------------------------------------------------------
	// stddev[j] = sqrt( (1/N) * sum_i (data[i][j] - mean[j])^2 )
	// stddev[j] = 1.0 if stddev[j] <= eps
	// ------------------------------------------------------------------------
	traverser(data, mean, stddev).template for_dims<'m'>([=](auto trav_m) {
		auto state_m = trav_m.state();

		stddev[state_m] = SCALAR_VAL(0.0);

		trav_m.for_each([=](auto state) {
			num_t val = data[state] - mean[state];
			stddev[state] += val * val;
		});

		stddev[state_m] /= float_n;
		stddev[state_m] = SQRT_FUN(stddev[state_m]);
		stddev[state_m] = stddev[state_m] <= eps ? SCALAR_VAL(1.0) : stddev[state_m];
	});

	// ------------------------------------------------------------------------
	// Center and reduce the column vectors:
	//   data[i][j] = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
	// ------------------------------------------------------------------------
	traverser(data, mean, stddev).template for_dims<'n'>([=](auto trav_n) {
		// trav_n has 'n' fixed; iterate over 'm'
		trav_n.template for_each<'m'>([=](auto state) {
			data[state] -= mean[state];
			data[state] /= SQRT_FUN(float_n) * stddev[state];
		});
	});

	// ------------------------------------------------------------------------
	// Compute the m x m correlation matrix:
	//
	// for (i = 0; i < M-1; i++) {
	//   corr[i][i] = 1.0;
	//   for (j = i+1; j < M; j++) {
	//     corr[i][j] = 0.0;
	//     for (k = 0; k < N; k++)
	//       corr[i][j] += data[k][i] * data[k][j];
	//     corr[j][i] = corr[i][j];
	//   }
	// }
	// corr[M-1][M-1] = 1.0;
	// ------------------------------------------------------------------------

	std::size_t m_len = corr | get_length<'p'>(); // M

	// Create views on data so that the feature index matches corr's 'p'/'q'
	auto data_p = data.get_ref() ^ noarr::rename<'m', 'p'>(); // dims: 'n', 'p'
	auto data_q = data.get_ref() ^ noarr::rename<'m', 'q'>(); // dims: 'n', 'q'

	// Outer loop over i = 0 .. M-1 (dimension 'p')
	traverser(corr).template for_dims<'p'>([=](auto trav_p) {
		auto state_p = trav_p.state();
		auto i = noarr::get_index<'p'>(state_p);

		// Only i < M-1 participate in the nested j-loop
		if (i < m_len - 1) {
			// corr[i][i] = 1.0;
			corr[noarr::idx<'p', 'q'>(i, i)] = SCALAR_VAL(1.0);

			// Middle loop over j = 0 .. M-1 (dimension 'q'), we only use j > i
			trav_p.template for_dims<'q'>([=](auto trav_pq) {
				auto state_pq = trav_pq.state();
				auto j = noarr::get_index<'q'>(state_pq);

				if (j > i) {
					// corr[i][j] = 0.0;
					corr[state_pq] = SCALAR_VAL(0.0);

					// Inner loop over k = 0 .. N-1 (dimension 'n')
					// Fix 'p' and 'q' for the data views to the current (i, j)
					noarr::traverser(data_p, data_q)
						.order(noarr::fix(state_pq))
						.for_each([=](auto state_n) {
							// state_n has 'n' (and fixed 'p','q' via fix)
							corr[state_pq] += data_p[state_n] * data_q[state_n];
						});

					// corr[j][i] = corr[i][j]; (symmetry)
					auto state_qp = noarr::idx<'p', 'q'>(j, i);
					corr[state_qp] = corr[state_pq];
				}
			});
		}
	});

	// Final diagonal element: corr[M-1][M-1] = 1.0;
	if (m_len > 0) {
		auto last = m_len - 1;
		corr[noarr::idx<'p', 'q'>(last, last)] = SCALAR_VAL(1.0);
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

	// set all relevant lengths
	auto set_lengths =
		noarr::set_length<'n'>(n) ^
		noarr::set_length<'m'>(m) ^
		noarr::set_length<'p'>(m) ^
		noarr::set_length<'q'>(m);

	// allocate bags for data, correlation matrix, mean, and stddev
	auto data   = noarr::bag(noarr::scalar<num_t>() ^ tuning.data_layout ^ set_lengths);
	auto corr   = noarr::bag(noarr::scalar<num_t>() ^ tuning.corr_layout ^ set_lengths);
	auto mean   = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout  ^ set_lengths);
	auto stddev = noarr::bag(noarr::scalar<num_t>() ^ tuning.vec_layout  ^ set_lengths);

	// initialize data
	init_array(float_n, data.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_correlation(float_n, data.get_ref(), corr.get_ref(), mean.get_ref(), stddev.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (correlation matrix) if requested
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'p' (row index) to top for a natural row-major printout
		noarr::serialize_data(std::cerr, corr.get_ref() ^ noarr::hoist<'p'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}