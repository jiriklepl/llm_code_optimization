#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "durbin.hpp"

using num_t = DATA_TYPE;

namespace {

// 1D proto-structure over dimension 'i'
constexpr auto i_vec = noarr::vector<'i'>();

struct tuning {
	// 1D layout over dimension 'i'
	DEFINE_PROTO_STRUCT(r_layout, i_vec);
	DEFINE_PROTO_STRUCT(y_layout, i_vec);
} tuning;

// initialization function: r[i] = n + 1 - i
void init_array(auto r) {
	using namespace noarr;

	auto n = r | get_length<'i'>();

	traverser(r).for_each([=](auto state) {
		auto i = get_index<'i'>(state);
		r[state] = static_cast<num_t>(n + 1 - i);
	});
}

// computation kernel
//
// Optimizations vs. original version:
//
//  - Removed the temporary array z[0..n-1] by performing the
//    z- and y-updates in-place using symmetric pairs (i, k-i-1).
//    For each i, the original code computes:
//        z[i] = y_old[i] + alpha * y_old[k - i - 1];
//        y[i] = z[i];
//    Our in-place update computes the same expression for each y[i],
//    but it processes indices in (i, j=k-i-1) pairs, first loading
//    both old values (a = y[i], b = y[j]) and then writing:
//        y[i] = a + alpha * b;
//        y[j] = b + alpha * a;
//    This uses only old values of y for the current k and thus
//    produces exactly the same per-element results as the original
//    z-based code, while halving memory traffic and improving cache
//    behavior.
//
//  - Kept all loop nests expressed via Noarr traversers, as required.
[[gnu::flatten, gnu::noinline]]
void kernel_durbin(auto r, auto y) {
	using namespace noarr;

	// length of the 1D vectors
	auto n = r | get_length<'i'>();

	// pseudo-structure for the k loop
	auto k_structure = noarr::scalar<char>() ^ noarr::vector<'k'>(n);

	num_t alpha;
	num_t beta;
	num_t sum;

	#pragma scop
	// y[0] = -r[0];
	y[noarr::idx<'i'>(0)] = -r[noarr::idx<'i'>(0)];
	beta = SCALAR_VAL(1.0);
	alpha = -r[noarr::idx<'i'>(0)];

	// for (k = 1; k < n; k++)
	noarr::traverser(k_structure)
		.order(noarr::slice<'k'>(1, n - 1))
		.for_each([&](auto k_state) {
			auto k = noarr::get_index<'k'>(k_state);

			// beta = (1 - alpha * alpha) * beta;
			beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

			// sum = 0.0;
			sum = SCALAR_VAL(0.0);

			// for (i = 0; i < k; i++) sum += r[k - i - 1] * y[i];
			noarr::traverser(y, r)
				.order(noarr::slice<'i'>(0, k))
				.for_each([&](auto ir_state) {
					auto i = noarr::get_index<'i'>(ir_state);

					auto r_state = noarr::idx<'i'>(k - i - 1);
					sum += r[r_state] * y[ir_state];
				});

			// alpha = - (r[k] + sum) / beta;
			alpha = -(r[noarr::idx<'i'>(k)] + sum) / beta;

			// In-place version of:
			//   for (i = 0; i < k; i++) {
			//     z[i] = y[i] + alpha * y[k - i - 1];
			//   }
			//   for (i = 0; i < k; i++) {
			//     y[i] = z[i];
			//   }
			//
			// We instead update y in-place using symmetric pairs of
			// indices (i, j = k - i - 1). For each pair we first load
			// old values a = y[i], b = y[j], then write:
			//   y[i] = a + alpha * b;
			//   y[j] = b + alpha * a;
			// This uses only old values and thus yields the same
			// final y as the original code with the temporary z.
			auto half_k = k / 2; // floor(k/2)

			if (half_k > 0) {
				noarr::traverser(y)
					.order(noarr::slice<'i'>(0, half_k))
					.for_each([&](auto i_state) {
						auto i = noarr::get_index<'i'>(i_state);
						auto j_state = noarr::idx<'i'>(k - i - 1);

						num_t y_i = y[i_state];
						num_t y_j = y[j_state];

						y[i_state] = y_i + alpha * y_j;
						y[j_state] = y_j + alpha * y_i;
					});
			}

			// If k is odd, there is one middle index m = (k-1)/2
			// that does not have a distinct partner. Original code:
			//   z[m] = y[m] + alpha * y[m];
			//   y[m] = z[m];
			// so we just do the equivalent in-place update.
			if (k % 2 != 0) {
				auto mid = (k - 1) / 2;
				auto mid_state = noarr::idx<'i'>(mid);
				num_t y_mid = y[mid_state];
				y[mid_state] = y_mid + alpha * y_mid;
			}

			// y[k] = alpha;
			y[noarr::idx<'i'>(k)] = alpha;
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// input data
	auto set_lengths = noarr::set_length<'i'>(n);

	auto r = noarr::bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ set_lengths);
	auto y = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout ^ set_lengths);

	// initialize data
	init_array(r.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_durbin(r.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (also prevents dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, y.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}