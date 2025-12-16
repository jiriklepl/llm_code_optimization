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

	// Iterate samples (k) outer, features (j) inner: row-major writes into `data`
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

	// Precompute normalization factors (used outside the hot inner loops)
	const num_t inv_n  = num_t(1) / float_n;
	const num_t inv_n1 = num_t(1) / (float_n - num_t(1));

#pragma scop
	// --------------------------------------------------------------------
	// 1) Compute column means: mean[j] = (1/n) * sum_k data[k][j]
	//
	// We do this in three sub-steps:
	//   a) zero-initialize mean[j]
	//   b) accumulate sums in a k-major, j-minor traversal
	//      (row-major over `data` for better spatial locality)
	//   c) scale by 1/n
	// --------------------------------------------------------------------

	// a) Initialize means to zero
	traverser(mean).for_each([&](auto s_mean) {
		mean[s_mean] = num_t(0);
	});

	// b) Accumulate sums row-by-row: outer over samples (k), inner over features (j)
	traverser(data, mean).template for_dims<'k'>([&](auto tk) {
		// `tk` has 'k' fixed; remaining dimension is 'j'
		tk.for_each([&](auto state) {
			// state has both 'k' and 'j'; `mean` ignores 'k' and uses only 'j'
			mean[state] += data[state];
		});
	});

	// c) Normalize each mean[j] by the number of samples
	traverser(mean).for_each([&](auto s_mean) {
		mean[s_mean] *= inv_n;
	});

	// --------------------------------------------------------------------
	// 2) Center data: data[k][j] -= mean[j]
	//
	// Traversal order is again k-major, j-minor so we walk rows of `data`
	// contiguously and reuse `mean[j]` from cache.
	// --------------------------------------------------------------------
	traverser(data, mean).template for_dims<'k'>([&](auto tk) {
		tk.for_each([&](auto state) {
			data[state] -= mean[state];
		});
	});

	// --------------------------------------------------------------------
	// 3) Covariance accumulation:
	//
	//    cov[i][j] = sum_k data[k][i] * data[k][j],   for all 0 <= i <= j < M
	//
	// We:
	//   - zero the whole covariance matrix,
	//   - then for each sample k,
	//       for each "row" feature i, load x_i = data[k][i] once,
	//       for each "column" feature j (upper triangle),
	//         accumulate x_i * data[k][j] into cov[i][j].
	//
	// Loop order: k (samples) -> i (row feature) -> j (column feature)
	// This yields:
	//   - row-major writes into cov (j is the contiguous dimension),
	//   - row-major reads from data (j is contiguous, k fixed),
	// and reuses x_i across all j for a given (k, i).
	// --------------------------------------------------------------------

	// Zero the covariance matrix
	traverser(cov).for_each([&](auto s_cov) {
		cov[s_cov] = num_t(0);
	});

	// Accumulate unnormalized covariance sums
	traverser(cov, data, data_i).template for_dims<'k', 'i'>([&](auto tki) {
		// Fixed sample k and row feature i
		auto s_ki = tki.state();          // contains 'k' and 'i'
		const num_t x_i = data_i[s_ki];   // x_i = data[k][i]

		// Now iterate over column feature j
		tki.template for_dims<'j'>([&](auto tkij) {
			auto s_kij = tkij.state();    // contains 'k', 'i', 'j'
			auto [i_idx, j_idx] = get_indices<'i', 'j'>(s_kij);

			// Only accumulate upper triangle (including diagonal)
			if (j_idx < i_idx)
				return;

			const num_t x_j = data[s_kij]; // x_j = data[k][j]

			cov[s_kij] += x_i * x_j;
		});
	});

	// --------------------------------------------------------------------
	// 4) Normalize by (n - 1) and enforce symmetry:
	//
	//     cov[i][j] = cov[j][i] = (1/(n-1)) * sum_k data[k][i] * data[k][j]
	//
	// We scale each upper-triangular element once and then mirror it
	// to the lower triangle.
	// --------------------------------------------------------------------
	traverser(cov).template for_dims<'i'>([&](auto ti) {
		ti.template for_dims<'j'>([&](auto tij) {
			auto s_ij = tij.state();
			auto [i_idx, j_idx] = get_indices<'i', 'j'>(s_ij);

			// We only process i <= j here
			if (j_idx < i_idx)
				return;

			// Scale the accumulated sum
			cov[s_ij] *= inv_n1;

			// Mirror to the symmetric position for off-diagonal elements
			if (j_idx != i_idx) {
				auto s_ji = noarr::idx<'i', 'j'>(j_idx, i_idx);
				cov[s_ji] = cov[s_ij];
			}
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
		// Print cov with 'i' as the hoisted (outer) dimension
		noarr::serialize_data(std::cerr, cov.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing to stdout
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}