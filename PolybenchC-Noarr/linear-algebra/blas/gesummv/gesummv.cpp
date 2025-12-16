#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions (DATA_TYPE, N, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "gesummv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

// layouts (row-major: inner dimension is the first one composed)
constexpr auto mat_layout = j_vec ^ i_vec; // i x j
constexpr auto x_layout   = j_vec;         // length N, indexed by 'j'
constexpr auto y_layout   = i_vec;         // length N, indexed by 'i'

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto x) {
	using namespace noarr;

	// matrices are N x N
	auto n = A | get_length<'i'>();

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	// Original C:
	// for (i = 0; i < n; i++) {
	//   x[i] = (DATA_TYPE)( i % n) / n;
	//   for (j = 0; j < n; j++) {
	//     A[i][j] = (DATA_TYPE)((i*j+1) % n) / n;
	//     B[i][j] = (DATA_TYPE)((i*j+2) % n) / n;
	//   }
	// }

	traverser(A, B).template for_dims<'i'>([=](auto inner) {
		// inner has 'i' fixed, 'j' remaining
		auto si = inner.state(); // contains index in 'i'
		auto i  = get_index<'i'>(si);

		// x[i]  — x is indexed by 'j', so we use j = i
		x[idx<'j'>(i)] = (num_t)(i % n) / n;

		// A[i][j], B[i][j]
		inner.for_each([=](auto s) {
			auto j = get_index<'j'>(s);

			A[s] = (num_t)((i * j + 1) % n) / n;
			B[s] = (num_t)((i * j + 2) % n) / n;
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y) {
	using namespace noarr;

	// Original C:
	// for (i = 0; i < _PB_N; i++) {
	//   tmp[i] = 0.0;
	//   y[i] = 0.0;
	//   for (j = 0; j < _PB_N; j++) {
	//     tmp[i] = A[i][j] * x[j] + tmp[i];
	//     y[i]   = B[i][j] * x[j] + y[i];
	//   }
	//   y[i] = alpha * tmp[i] + beta * y[i];
	// }

	#pragma scop
	traverser(A, B, tmp, x, y).template for_dims<'i'>([=](auto inner) {
		// inner has 'i' fixed, 'j' remaining
		auto si = inner.state(); // state with fixed 'i'

		// tmp[i] = 0; y[i] = 0;
		tmp[si] = (num_t)0.0;
		y[si]   = (num_t)0.0;

		// for j: accumulate tmp[i] and y[i]
		inner.for_each([=](auto s) {
			// s has both 'i' and 'j'
			auto j = get_index<'j'>(s);

			// x[j]
			auto sj = idx<'j'>(j);

			tmp[si] = A[s] * x[sj] + tmp[si];
			y[si]   = B[s] * x[sj] + y[si];
		});

		// y[i] = alpha * tmp[i] + beta * y[i];
		y[si] = alpha * tmp[si] + beta * y[si];
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;
	using namespace noarr;

	// problem size
	std::size_t n = N;

	// scalars
	num_t alpha;
	num_t beta;

	// common lengths for dimensions 'i' and 'j'
	auto set_lengths =
		set_length<'i'>(n) ^
		set_length<'j'>(n);

	// data structures
	auto A   = bag(scalar<num_t>() ^ mat_layout ^ set_lengths);
	auto B   = bag(scalar<num_t>() ^ mat_layout ^ set_lengths);
	auto tmp = bag(scalar<num_t>() ^ y_layout   ^ set_lengths); // length N in 'i'
	auto x   = bag(scalar<num_t>() ^ x_layout   ^ set_lengths); // length N in 'j'
	auto y   = bag(scalar<num_t>() ^ y_layout   ^ set_lengths); // length N in 'i'

	// initialize data
	init_array(alpha, beta, A.get_ref(), B.get_ref(), x.get_ref());

	// timing
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gesummv(alpha, beta,
	               A.get_ref(), B.get_ref(),
	               tmp.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// optional: print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		serialize_data(std::cerr, y.get_ref() ^ hoist<'i'>());
	}

	// print elapsed time
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}