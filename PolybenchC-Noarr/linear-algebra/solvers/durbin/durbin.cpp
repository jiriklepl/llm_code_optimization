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
[[gnu::flatten, gnu::noinline]]
void kernel_durbin(auto r, auto y) {
	using namespace noarr;

	// length of the 1D vectors
	auto n = r | get_length<'i'>();

	// local temporary z[0..n-1]
	auto z = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));

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
		.template for_dims<'k'>([&](auto k_trav) {
			auto k_state = k_trav.state();
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

			// for (i = 0; i < k; i++) z[i] = y[i] + alpha * y[k - i - 1];
			noarr::traverser(y, z)
				.order(noarr::slice<'i'>(0, k))
				.for_each([&](auto iz_state) {
					auto i = noarr::get_index<'i'>(iz_state);

					auto y_rev_state = noarr::idx<'i'>(k - i - 1);
					z[iz_state] = y[iz_state] + alpha * y[y_rev_state];
				});

			// for (i = 0; i < k; i++) y[i] = z[i];
			noarr::traverser(y, z)
				.order(noarr::slice<'i'>(0, k))
				.for_each([&](auto iz_state) {
					y[iz_state] = z[iz_state];
				});

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