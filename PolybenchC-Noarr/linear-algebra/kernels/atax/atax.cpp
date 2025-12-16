#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "atax.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A: m x n, row-major (i = row, j = column)
	DEFINE_PROTO_STRUCT(A_layout, j_vec ^ i_vec);
	// x: length n (dimension 'j')
	DEFINE_PROTO_STRUCT(x_layout, j_vec);
	// y: length n (dimension 'j')
	DEFINE_PROTO_STRUCT(y_layout, j_vec);
	// tmp: length m (dimension 'i')
	DEFINE_PROTO_STRUCT(tmp_layout, i_vec);
} tuning;

// Array initialization
void init_array(auto A, auto x) {
	using namespace noarr;

	// x: j dimension of length N
	auto n_len = x | get_length<'j'>();
	num_t fn = static_cast<num_t>(n_len);

	// x[j] = 1 + (j / fn)
	traverser(x).for_each([=](auto state) {
		auto j = get_index<'j'>(state);
		x[state] = static_cast<num_t>(1) + static_cast<num_t>(j) / fn;
	});

	// A: dimensions i (M) and j (N)
	auto m_len = A | get_length<'i'>();
	auto n_len_A = A | get_length<'j'>();

	// A[i][j] = ((i + j) % n) / (5 * m)
	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		num_t numerator = static_cast<num_t>((i + j) % n_len_A);
		num_t denom = static_cast<num_t>(5 * m_len);
		A[state] = numerator / denom;
	});
}

// Main computational kernel
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) {
	using namespace noarr;

	#pragma scop
	// for (i = 0; i < N; i++) y[i] = 0;
	traverser(y).for_each([=](auto sy) {
		y[sy] = static_cast<num_t>(0);
	});

	// for (i = 0; i < M; i++) { ... }
	traverser(A, x, y, tmp).template for_dims<'i'>([=](auto inner) {
		// state with fixed i
		auto si = inner.state();

		// tmp[i] = 0;
		tmp[si] = static_cast<num_t>(0);

		// for (j = 0; j < N; j++)
		//   tmp[i] = tmp[i] + A[i][j] * x[j];
		inner.template for_each<'j'>([=](auto state) {
			tmp[si] += A[state] * x[state];
		});

		// for (j = 0; j < N; j++)
		//   y[j] = y[j] + A[i][j] * tmp[i];
		inner.template for_each<'j'>([=](auto state) {
			y[state] += A[state] * tmp[si];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t m = M;
	std::size_t n = N;

	// common length proto-structure
	auto set_lengths = noarr::set_length<'i'>(m) ^ noarr::set_length<'j'>(n);

	// data structures
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.A_layout   ^ set_lengths);
	auto x   = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout   ^ set_lengths);
	auto y   = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout   ^ set_lengths);
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), x.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print result vector y
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, y.get_ref());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}