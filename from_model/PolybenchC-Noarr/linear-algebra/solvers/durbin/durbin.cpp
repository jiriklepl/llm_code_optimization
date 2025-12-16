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
		r[state] = (num_t)(n + 1 - i);
	});
}

// computation kernel
//
// Optimizations compared to the original version:
//   - Remove the temporary vector z and perform the update of y in-place
//     using symmetric pairs (i, j = k-1-i). This preserves the exact
//     mathematical behavior of the original two-loop z-based formulation:
//         z[i] = y[i] + alpha * y[k-1-i];
//         y[i] = z[i];
//     but halves the number of passes over y and avoids allocating / touching z.
//   - Keep all loops expressed via Noarr traversers; i- and k-loops remain
//     1D traversals over their respective dimensions.
[[gnu::flatten, gnu::noinline]]
void kernel_durbin(auto r, auto y) {
	using namespace noarr;

	// length of the 1D vectors
	auto n = r | get_length<'i'>();

	// pseudo-structure for the k loop (dimension 'k' from 0 to n-1)
	auto k_structure = noarr::scalar<char>() ^ noarr::vector<'k'>(n);

	num_t alpha;
	num_t beta;
	num_t sum;

	#pragma scop
	// Initial step:
	//   y[0] = -r[0];
	//   beta = 1.0;
	//   alpha = -r[0];
	y[noarr::idx<'i'>(0)] = -r[noarr::idx<'i'>(0)];
	beta = SCALAR_VAL(1.0);
	alpha = -r[noarr::idx<'i'>(0)];

	// Main Durbin loop: for (k = 1; k < n; k++)
	//
	// We traverse k using a Noarr traverser over k_structure, slicing
	// away k = 0 (already handled above).
	noarr::traverser(k_structure)
		.order(noarr::slice<'k'>(1, n - 1))
		.for_each([&](auto k_state) {
			auto k = noarr::get_index<'k'>(k_state);

			// beta = (1 - alpha * alpha) * beta;
			beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

			// sum = 0.0;
			sum = SCALAR_VAL(0.0);

			// sum += r[k - i - 1] * y[i] for i in [0, k)
			//
			// This is a dot product between:
			//   y[0..k-1] in forward order and
			//   r[0..k-1] in reverse order (r[k-1-i]).
			// We express the i-loop with a Noarr traverser over y and r
			// restricted to the prefix [0, k).
			noarr::traverser(y, r)
				.order(noarr::slice<'i'>(0, k))
				.for_each([&](auto ir_state) {
					auto i = noarr::get_index<'i'>(ir_state);
					auto r_state = noarr::idx<'i'>(k - i - 1);
					sum += r[r_state] * y[ir_state];
				});

			// alpha_k = - (r[k] + sum) / beta;
			num_t alpha_new = -(r[noarr::idx<'i'>(k)] + sum) / beta;

			// In-place symmetric update of y[0..k-1], eliminating z:
			//
			// Original code:
			//   for (i = 0; i < k; i++) z[i] = y[i] + alpha * y[k - i - 1];
			//   for (i = 0; i < k; i++) y[i] = z[i];
			//
			// Let y_old be the vector before these loops and y_new after them.
			// Algebraically:
			//   y_new[i] = y_old[i] + alpha * y_old[k-1-i]
			//
			// We can compute this without a temporary z by updating symmetric
			// index pairs (i, j = k-1-i) once, using their old values:
			//   y_i_new = y_i_old + alpha_new * y_j_old
			//   y_j_new = y_j_old + alpha_new * y_i_old
			// For odd k there is a middle element m = (k-1)/2 with i == j:
			//   y_m_new = y_m_old + alpha_new * y_m_old.
			//
			// Here we process only i in [0, floor(k/2)), so j = k-1-i always
			// satisfies j > i and we never overwrite a value that is still
			// needed as "old". For the middle element (when k is odd) we do
			// a single scalar update afterwards.
			std::size_t half = k / 2;

			noarr::traverser(y)
				.order(noarr::slice<'i'>(0, half))
				.for_each([&](auto yi_state) {
					auto i = noarr::get_index<'i'>(yi_state);
					std::size_t j = k - 1 - i;

					// i ∈ [0, half), j ∈ (half-1 .. k-1], so i < j always holds.
					auto y_i_old = y[yi_state];
					auto y_j_state = noarr::idx<'i'>(j);
					auto y_j_old = y[y_j_state];

					y[yi_state]  = y_i_old + alpha_new * y_j_old;
					y[y_j_state] = y_j_old + alpha_new * y_i_old;
				});

			// Middle element when k is odd: m = (k-1)/2, with i == j.
			// We must update it once using its old value:
			//   y[m] = y_old[m] + alpha_new * y_old[m].
			if (k % 2 == 1) {
				std::size_t m = (k - 1) / 2;
				auto y_m_state = noarr::idx<'i'>(m);
				auto y_m_old = y[y_m_state];
				y[y_m_state] = y_m_old + alpha_new * y_m_old;
			}

			// y[k] = alpha_k;
			y[noarr::idx<'i'>(k)] = alpha_new;

			// Carry alpha_k to the next iteration.
			alpha = alpha_new;
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