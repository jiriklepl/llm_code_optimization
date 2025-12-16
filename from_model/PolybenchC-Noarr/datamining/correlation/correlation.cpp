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
	// data: n x m  (rows: n, columns/features: m), column-major (n contiguous)
	// Layout: scalar ^ n_vec ^ m_vec  => dimensions: 'm' (outer), 'n' (inner, contiguous)
	DEFINE_PROTO_STRUCT(data_layout, n_vec ^ m_vec);

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

	// M is the length in the 'm' dimension; compute once outside the loop
	const num_t M_len = static_cast<num_t>(data | get_length<'m'>());

	noarr::traverser(data).for_each([=](auto state) {
		auto [i, j] = get_indices<'n', 'm'>(state);

		data[state] =
			static_cast<num_t>(static_cast<num_t>(i * j) / M_len + static_cast<num_t>(i));
	});
}


// main computational kernel
[[gnu::flatten, gnu::noinline]]
void kernel_correlation(num_t float_n, auto data, auto corr, auto mean, auto stddev) {
	using namespace noarr;

	const num_t eps = SCALAR_VAL(0.1);

	// Precompute scalar factors used in all phases (do this outside the SCoP).
	const num_t inv_float_n = SCALAR_VAL(1.0) / float_n; // 1/N
	const num_t sqrt_n      = SQRT_FUN(float_n);         // sqrt(N)
	const num_t inv_sqrt_n  = SCALAR_VAL(1.0) / sqrt_n;  // 1 / sqrt(N)

#pragma scop

	// ------------------------------------------------------------------------
	// 1. Column means:
	//    mean[m] = (1/N) * sum_n data[n][m]
	//
	// With column-major layout (n contiguous for fixed m), we fix 'm' and
	// iterate 'n' as the inner dimension to get unit-stride access.
	// ------------------------------------------------------------------------
	noarr::traverser(data, mean).template for_dims<'m'>([&](auto trav_m) {
		// trav_m has 'm' fixed; 'n' remains to iterate over rows.
		auto state_m = trav_m.state();

		num_t sum = SCALAR_VAL(0.0);

		// Walk contiguous elements along 'n' for this feature column.
		trav_m.template for_each<'n'>([&](auto state) {
			sum += data[state];
		});

		mean[state_m] = sum * inv_float_n;
	});

	// ------------------------------------------------------------------------
	// 2. Column standard deviations:
	//    stddev[m] = sqrt( (1/N) * sum_n (data[n,m] - mean[m])^2 )
	//    if stddev[m] <= eps then stddev[m] = 1.0
	//
	// Again, exploit column-major layout by fixing 'm' and streaming in 'n'.
	// ------------------------------------------------------------------------
	noarr::traverser(data, mean, stddev).template for_dims<'m'>([&](auto trav_m) {
		auto state_m = trav_m.state();

		const num_t mean_val = mean[state_m];
		num_t acc = SCALAR_VAL(0.0);

		trav_m.template for_each<'n'>([&](auto state) {
			const num_t diff = data[state] - mean_val;
			acc += diff * diff;
		});

		num_t sigma2 = acc * inv_float_n;
		num_t sigma  = SQRT_FUN(sigma2);
		sigma = sigma <= eps ? SCALAR_VAL(1.0) : sigma;

		stddev[state_m] = sigma;
	});

	// ------------------------------------------------------------------------
	// 3. Center and normalize each column in-place:
	//    Original:
	//      data[n,m] = (data[n,m] - mean[m]) / (sqrt(float_n) * stddev[m])
	//    Using precomputed inv_sqrt_n:
	//      data[n,m] = (data[n,m] - mean[m]) * (inv_sqrt_n / stddev[m])
	//
	// This keeps the math identical but avoids a sqrt per element and
	// replaces the divide by a multiply plus one divide per column.
	// ------------------------------------------------------------------------
	noarr::traverser(data, mean, stddev).template for_dims<'m'>([&](auto trav_m) {
		auto state_m = trav_m.state();

		const num_t mean_val = mean[state_m];
		const num_t sigma    = stddev[state_m];
		const num_t factor   = inv_sqrt_n / sigma; // one division per feature

		trav_m.template for_each<'n'>([&](auto state) {
			data[state] = (data[state] - mean_val) * factor;
		});
	});

	// ------------------------------------------------------------------------
	// 4. Compute the M x M correlation matrix:
	//
	// For normalized data:
	//   corr[p,p] = 1.0
	//   corr[p,q] = sum_n data[n,p] * data[n,q]   for q > p
	//   corr[q,p] = corr[p,q]                     (symmetry)
	//
	// We keep corr row-major (q contiguous), but build read-only views
	// of `data` where the feature index matches 'p'/'q'. With the chosen
	// column-major layout, each column data[*,p] and data[*,q] is contiguous
	// in 'n', so the inner dot-product has unit-stride accesses.
	// ------------------------------------------------------------------------

	// Views on data: map feature index 'm' to 'p' or 'q' used by corr
	auto data_p = data.get_ref() ^ noarr::rename<'m', 'p'>(); // dims: 'p', 'n'
	auto data_q = data.get_ref() ^ noarr::rename<'m', 'q'>(); // dims: 'q', 'n'

	// Outer loop over p = 0 .. M-1 (rows of corr)
	noarr::traverser(corr).template for_dims<'p'>([&](auto trav_p) {
		auto state_p = trav_p.state();
		const std::size_t i = noarr::get_index<'p'>(state_p);

		// Diagonal element corr[i][i] = 1.0
		corr[noarr::idx<'p', 'q'>(i, i)] = SCALAR_VAL(1.0);

		// Inner loop over q = 0 .. M-1, but we only use q > i (upper triangle)
		trav_p.template for_dims<'q'>([&](auto trav_pq) {
			auto state_pq = trav_pq.state();
			const std::size_t j = noarr::get_index<'q'>(state_pq);

			if (j <= i)
				return; // skip lower triangle and diagonal (already set)

			num_t acc = SCALAR_VAL(0.0);

			// Dot product of columns i and j over all rows 'n'.
			noarr::traverser(data_p, data_q)
				.order(noarr::fix(state_pq))          // fix p=i, q=j
				.template for_each<'n'>([&](auto state_n) {
					acc += data_p[state_n] * data_q[state_n];
				});

			corr[state_pq] = acc;
			// Enforce symmetry: corr[j][i] = corr[i][j]
			corr[noarr::idx<'p', 'q'>(j, i)] = acc;
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