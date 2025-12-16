#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions (DATA_TYPE, N, DEFINE_PROTO_STRUCT, ...)
#include "defines.hpp"

// include benchmark-specific definitions (for trisolv)
#include "trisolv.hpp"

using num_t = DATA_TYPE;

namespace {

// dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

// layouts for individual arrays
struct tuning {
	// L: i x j  (row i, column j)
	DEFINE_PROTO_STRUCT(l_layout, j_vec ^ i_vec);
	// x: i
	DEFINE_PROTO_STRUCT(x_layout, i_vec);
	// b: i
	DEFINE_PROTO_STRUCT(b_layout, i_vec);
} tuning;

// Array initialization.
// L: n x n lower triangular
// x: length n
// b: length n
void init_array(auto L, auto x, auto b) {
	using namespace noarr;

	// problem size from structure
	auto n = L | get_length<'i'>();

	// initialize x and b: for i in [0, n)
	traverser(x, b).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x[state] = (num_t)-999;
		b[state] = (num_t)i;
	});

	// initialize L: for i in [0, n), for j in [0, n), but only j <= i are set
	traverser(L).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		if (j <= i) {
			L[state] = (num_t)(i + n - j + 1) * 2 / n;
		}
	});
}

// Main computational kernel:
// for i = 0..n-1:
//   x[i] = b[i];
//   for j = 0..i-1:
//     x[i] -= L[i][j] * x[j];
//   x[i] = x[i] / L[i][i];
[[gnu::flatten, gnu::noinline]]
void kernel_trisolv(auto L, auto x, auto b) {
	using namespace noarr;

#pragma scop
	// traverse in the 'i' dimension outermost, as in the original code
	traverser(L, x, b).template for_dims<'i'>([=](auto inner) {
		// state with current i fixed
		auto state_i = inner.state();
		auto i = get_index<'i'>(state_i);

		// x[i] = b[i];
		x[state_i] = b[state_i];

		// for j = 0; j < i; j++
		inner.template for_each<'j'>([=](auto state) {
			auto j = get_index<'j'>(state);

			if (j < i) {
				// L[i][j] is indexed by (i, j) in 'state'
				// x[j] is indexed by 'i' = j
				x[state_i] -= L[state] * x[noarr::idx<'i'>(j)];
			}
		});

		// x[i] = x[i] / L[i][i];
		auto diag_state = noarr::idx<'i', 'j'>(i, i);
		x[state_i] = x[state_i] / L[diag_state];
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// set lengths for dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// allocate data (bags own the memory)
	auto L = noarr::bag(noarr::scalar<num_t>() ^ tuning.l_layout ^ set_lengths);
	auto x = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto b = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize arrays
	init_array(L.get_ref(), x.get_ref(), b.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trisolv(L.get_ref(), x.get_ref(), b.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (prevents dead-code elimination of computation)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x.get_ref());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}