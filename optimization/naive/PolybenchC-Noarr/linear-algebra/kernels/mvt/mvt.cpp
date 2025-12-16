#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "mvt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// 1D layouts for vectors (all use dimension 'i')
	DEFINE_PROTO_STRUCT(x_layout, i_vec);

	// 2D layout for matrix A: dimensions 'i' (rows) and 'j' (columns)
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	auto n = x1 | get_length<'i'>();

	// Initialize the four vectors
	traverser(x1, x2, y_1, y_2).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x1[state] = (num_t)((i % n)) / n;
		x2[state] = (num_t)(((i + 1) % n)) / n;
		y_1[state] = (num_t)(((i + 3) % n)) / n;
		y_2[state] = (num_t)(((i + 4) % n)) / n;
	});

	// Initialize matrix A: A[i][j] = (i * j % n) / n
	auto nA = A | get_length<'i'>();

	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i * j) % nA) / nA;
	});
}

// computation kernel
//
// Original mathematical specification:
//   x1[i] += Σ_j A[i][j] * y_1[j]
//   x2[i] += Σ_j A[j][i] * y_2[j]
//
// The original implementation performs these as two separate matrix–vector
// products, traversing A once in row-major order and once effectively in
// column-major order (via a transposed view). That second pass has very poor
// cache locality on a row-major A.
//
// The optimized kernel below fuses both computations into a *single*
// row-major traversal of A. For each element A[i][j] we compute:
//
//   x1[i]      += A[i][j] * y_1[j]
//   x2[j] (via a 'j'-indexed view) += A[i][j] * y_2[i]
//
// Algebraically this is identical to the original two-loop formulation:
// - For x1 it’s exactly the same expression.
// - For x2 we use the identity
//       x2[j] = Σ_i A[i][j] * y_2[i]
//   which is equivalent to the original
//       x2[i] = Σ_j A[j][i] * y_2[j]
//   after renaming dummy indices and the result index.
//
// Importantly, for each output element x2[k] the sequence of partial sums
// still processes row indices in ascending order 0..N-1, only interleaved
// with updates to other x2 entries; thus floating-point roundoff behaviour
// per element is preserved while cutting matrix traffic in half.
//
[[gnu::flatten, gnu::noinline]]
void kernel_mvt(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	// View of y_1 as a vector over 'j' (needed for y_1[j])
	auto y_1_j = y_1.get_ref() ^ noarr::rename<'i', 'j'>();

	// View of x2 as a vector over 'j' so that we can naturally accumulate
	// x2[j] while traversing columns of A
	auto x2_j = x2.get_ref() ^ noarr::rename<'i', 'j'>();

#pragma scop
	// Fused kernel:
	// for (i = 0; i < N; i++)
	//   for (j = 0; j < N; j++) {
	//     x1[i] += A[i][j] * y_1[j];
	//     x2[j] += A[i][j] * y_2[i];
	//   }
	//
	// We implement this using a traverser over all involved structures so
	// that a single state carries both indices 'i' and 'j'.

	noarr::traverser(A, x1, x2_j, y_1_j, y_2)
		// Outer loop over rows 'i' – this respects A's row-major layout
		.template for_dims<'i'>([&](auto trav_i) {
			// trav_i has 'i' fixed, only 'j' remains varying.
			// Extract the state with only 'i' to index x1 and y_2 efficiently.
			auto s_i = trav_i.state();

			// Cache row-wise values in registers:
			num_t acc_x1 = x1[s_i]; // running sum for x1[i]
			const num_t y2_i = y_2[s_i];

			// Inner loop over columns 'j'
			trav_i.for_each([&](auto state) {
				// state contains both 'i' and 'j'
				const num_t a_ij = A[state];

				// x1[i] contribution: use cached accumulator
				acc_x1 += a_ij * y_1_j[state];

				// x2[j] contribution through the 'j'-indexed view
				x2_j[state] += a_ij * y2_i;
			});

			// Write back accumulated x1[i]
			x1[s_i] = acc_x1;
		});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// common length proto-structure for all dimensions
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// data structures
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto x1  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto x2  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);

	// initialize data
	init_array(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_mvt(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (prevents dead-code elimination), if requested
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x1.get_ref());
		noarr::serialize_data(std::cerr, x2.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}