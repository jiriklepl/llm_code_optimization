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

	// Precompute M once instead of querying inside the loop
	const num_t M_len = static_cast<num_t>(data | get_length<'m'>());

	// Default traverser order for data (dims 'n','m') is row-major (m contiguous)
	traverser(data).for_each([=](auto state) {
		auto [i, j] = get_indices<'n', 'm'>(state);

		data[state] =
			static_cast<num_t>(static_cast<num_t>(i * j) / M_len + static_cast<num_t>(i));
	});
}


// main computational kernel
// Optimized for:
//   - row-major access to `data` ('n' outer, 'm' inner)
//   - reduced redundant sqrt/division operations
//   - minimal structure / bag queries inside hot loops
[[gnu::flatten, gnu::noinline]]
void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
	using namespace noarr;

	const num_t eps        = SCALAR_VAL(0.1);
	const num_t inv_float_n = SCALAR_VAL(1.0) / float_n;
	const num_t sqrt_n      = SQRT_FUN(float_n);

#pragma scop

	// ------------------------------------------------------------------------
	// mean[j] = 1/N * sum_i data[i][j]
	// ------------------------------------------------------------------------

	// Initialize mean[:] = 0
	traverser(mean).for_each([=](auto state_m) {
		mean[state_m] = SCALAR_VAL(0.0);
	});

	// Row-major accumulation: outer 'n' (rows), inner 'm' (features).
	// This matches the physical layout of `data` (m contiguous), improving
	// cache locality compared to the original column-wise accumulation.
	traverser(data, mean).template for_dims<'n'>([=](auto trav_n) {
		// trav_n has 'n' fixed; iterate contiguous 'm' for this row
		trav_n.template for_each<'m'>([=](auto state_nm) {
			// state_nm has both 'n' and 'm'; mean ignores 'n'
			mean[state_nm] += data[state_nm];
		});
	});

	// Scale by 1/N
	traverser(mean).for_each([=](auto state_m) {
		mean[state_m] *= inv_float_n;
	});


	// ------------------------------------------------------------------------
	// stddev[j] = sqrt( (1/N) * sum_i (data[i][j] - mean[j])^2 )
	// stddev[j] = 1.0 if stddev[j] <= eps
	//
	// Optimization:
	//   We compute variance in a row-major pass, then:
	//     sigma_j = max(stddev[j], eps)  (with eps rule)
	//     stddev[j] <- 1 / (sqrt(N) * sigma_j)
	//   i.e. we turn stddev[] into per-feature scaling factors used below.
	// ------------------------------------------------------------------------

	// Initialize stddev[:] = 0
	traverser(stddev).for_each([=](auto state_m) {
		stddev[state_m] = SCALAR_VAL(0.0);
	});

	// Accumulate squared deviations, again in row-major order
	traverser(data, mean, stddev).template for_dims<'n'>([=](auto trav_n) {
		trav_n.template for_each<'m'>([=](auto state_nm) {
			const num_t val = data[state_nm] - mean[state_nm];
			stddev[state_nm] += val * val;
		});
	});

	// Finish stddev computation and overwrite it with
	//   inv_scale[j] = 1 / (sqrt(N) * stddev[j])
	// so that the normalization step becomes just:
	//   data[i][j] = (data[i][j] - mean[j]) * inv_scale[j]
	traverser(stddev).for_each([=](auto state_m) {
		stddev[state_m] *= inv_float_n;      // variance
		num_t sigma = SQRT_FUN(stddev[state_m]);
		if (sigma <= eps) sigma = SCALAR_VAL(1.0);
		stddev[state_m] = SCALAR_VAL(1.0) / (sqrt_n * sigma);
	});


	// ------------------------------------------------------------------------
	// Center and reduce the column vectors:
	//   data[i][j] = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
	//              = (data[i][j] - mean[j]) * (1 / (sqrt_n * stddev[j]))
	//
	// This uses the precomputed per-feature scaling factors in stddev[].
	// Traversal: row-major (n outer, m inner) for good locality.
	// ------------------------------------------------------------------------
	traverser(data, mean, stddev).template for_dims<'n'>([=](auto trav_n) {
		// trav_n has 'n' fixed; iterate over 'm'
		trav_n.template for_each<'m'>([=](auto state_nm) {
			data[state_nm] = (data[state_nm] - mean[state_nm]) * stddev[state_nm];
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
	//
	// We preserve this computation order, but:
	//   - we zero the whole matrix once up front
	//   - we avoid rebuilding traversers for each (i,j) by
	//     using a joint traverser(corr, data_p, data_q) and
	//     iterating 'n' via for_each<'n'>().
	// ------------------------------------------------------------------------

	const std::size_t m_len = corr | get_length<'p'>(); // M

	// Views on data so that feature index matches corr's 'p'/'q'
	auto data_p = data.get_ref() ^ noarr::rename<'m', 'p'>(); // dims: 'n', 'p'
	auto data_q = data.get_ref() ^ noarr::rename<'m', 'q'>(); // dims: 'n', 'q'

	// Initialize corr to 0.0 (upper + lower triangle)
	traverser(corr).for_each([=](auto state_pq) {
		corr[state_pq] = SCALAR_VAL(0.0);
	});

	// Outer loop over i = 0 .. M-1 (dimension 'p')
	traverser(corr, data_p, data_q).template for_dims<'p'>([=](auto trav_p) {
		const auto state_p = trav_p.state();
		const std::size_t i = noarr::get_index<'p'>(state_p);

		// Only i < M-1 participate in the nested j-loop
		if (i < m_len - 1) {
			// Middle loop over j = 0 .. M-1 (dimension 'q'), we only use j > i
			trav_p.template for_dims<'q'>([=](auto trav_pq) {
				const auto state_pq = trav_pq.state();
				const std::size_t j = noarr::get_index<'q'>(state_pq);

				if (j > i) {
					// Accumulate corr[i][j] over k = 0 .. N-1 (dimension 'n')
					trav_pq.template for_each<'n'>([=](auto state_n) {
						// state_n has 'n' plus fixed 'p','q' via the outer traversers
						corr[state_pq] += data_p[state_n] * data_q[state_n];
					});

					// corr[j][i] = corr[i][j]; (symmetry)
					auto state_qp = noarr::idx<'p', 'q'>(j, i);
					corr[state_qp] = corr[state_pq];
				}
			});

			// corr[i][i] = 1.0;
			corr[noarr::idx<'p', 'q'>(i, i)] = SCALAR_VAL(1.0);
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