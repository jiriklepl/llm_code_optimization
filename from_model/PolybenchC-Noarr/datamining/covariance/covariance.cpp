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

	// Precompute reciprocals to avoid repeated divisions in inner loops.
	const num_t inv_float_n    = num_t(1.0) / float_n;
	const num_t inv_n_minus_1  = num_t(1.0) / (float_n - num_t(1.0));

	// view of data where the feature dimension is called 'i'
	// (for cov's row index); data_i: k x i
	// Note: this is a pure re-interpretation of the layout, no data is copied.
	auto data_i = data ^ noarr::rename<'j', 'i'>();

#pragma scop
	// ------------------------------------------------------------
	// 1) Compute per-feature means:
	//      mean[j] = (1 / float_n) * sum_k data[k][j]
	//
	// The original code iterated j outer, k inner, which walks `data`
	// in a strided pattern (column-wise) because `j` is the innermost
	// contiguous dimension in the layout j_vec ^ k_vec.
	//
	// Here we switch to k-outer / j-inner:
	//   for k:
	//     for j:
	//       mean[j] += data[k,j]
	//
	// which streams row-by-row through `data` and keeps `mean` hot
	// in cache. All arithmetic is equivalent, only the reduction
	// order over k changes.
	// ------------------------------------------------------------

	// 1a) mean[j] = 0
	traverser(mean).for_each([=](auto sj) {
		mean[sj] = num_t(0);
	});

	// 1b) accumulate sums over samples, streaming across rows
	traverser(data, mean).template for_dims<'k'>([=](auto tk) {
		tk.template for_each<'j'>([=](auto state) {
			// state has ('k','j'); `mean` ignores extra dimensions
			mean[state] += data[state];
		});
	});

	// 1c) scale by 1 / N
	traverser(mean).for_each([=](auto sj) {
		mean[sj] *= inv_float_n;
	});

	// ------------------------------------------------------------
	// 2) Center the data in-place:
	//      data[k][j] -= mean[j]
	//
	// We keep the original k-outer / j-inner traversal, which is
	// already cache- and SIMD‑friendly for our row-major layout.
	// ------------------------------------------------------------
	traverser(data, mean).template for_dims<'k'>([=](auto tk) {
		tk.template for_each<'j'>([=](auto state) {
			data[state] -= mean[state]; // `mean` uses only 'j'
		});
	});

	// ------------------------------------------------------------
	// 3) Covariance:
	//
	// After centering, we want:
	//   for 0 <= i <= j < M:
	//     cov[i][j] = (1 / (float_n - 1)) * sum_k data[k,i] * data[k,j]
	//     cov[j][i] = cov[i][j]
	//
	// This is equivalent to cov = Xᵀ X / (N-1) with X = centered data.
	//
	// Compared to the original i‑j‑k ordering (which scanned k with a
	// large stride), we change the reduction order to:
	//
	//   for k:
	//     for i:
	//       x_ki = data[k,i]
	//       for j >= i:
	//         cov[i,j] += x_ki * data[k,j]
	//
	// This keeps the inner loop over `j` contiguous in memory and
	// loads each data[k,i] only once per (k,i) pair. The number of
	// arithmetic operations is unchanged; only the traversal order
	// and memory access pattern differ.
	// ------------------------------------------------------------

	// 3a) initialize covariance matrix to zero
	traverser(cov).for_each([=](auto s) {
		cov[s] = num_t(0);
	});

	// 3b) accumulate outer products sample-by-sample
	traverser(cov, data, data_i).template for_dims<'k'>([=](auto tk) {
		// k is fixed here (one sample row)

		// iterate over all covariance rows i
		tk.template for_dims<'i'>([=](auto tki) {
			// State with dimensions {k, i} (and possibly others ignored by data_i)
			auto si = tki.state();

			// Load data[k][i] once and reuse across all j in this (k, i) slice
			const num_t x_ki = data_i[si];

			// iterate over covariance columns j
			tki.template for_dims<'j'>([=](auto tkij) {
				auto s = tkij.state(); // has at least 'i' and 'j' (and 'k')

				auto [i_idx, j_idx] = get_indices<'i', 'j'>(s);
				// exploit symmetry: compute only upper triangle (including diagonal)
				if (j_idx < i_idx)
					return;

				// data[k][j]: contiguous across j for fixed k
				const num_t x_kj = data[s];

				// s contains extra 'k' dimension, which is ignored by `cov`
				cov[s] += x_ki * x_kj;
			});
		});
	});

	// 3c) normalize by (N-1) and explicitly symmetrize:
	//       cov[i,j] /= (float_n - 1)
	//       cov[j,i] = cov[i,j]
	traverser(cov).template for_dims<'i'>([=](auto ti) {
		ti.template for_dims<'j'>([=](auto tij) {
			auto s = tij.state();
			auto [i_idx, j_idx] = get_indices<'i', 'j'>(s);

			// only upper triangle (including diagonal) is the primary computation
			if (j_idx < i_idx)
				return;

			// scale accumulated sum
			cov[s] *= inv_n_minus_1;

			// enforce symmetry: cov[j][i] = cov[i][j]
			auto s_ji = noarr::idx<'i', 'j'>(j_idx, i_idx);
			cov[s_ji] = cov[s];
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