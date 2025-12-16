#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "symm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// C: i x j, row-major (j is the innermost dimension)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);

	// A: i x k (M x M), row-major in k (k is the innermost dimension)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);

	// B: i x j (M x N), **column-major**: i is the innermost dimension
	// This makes B[k, j] with k varying contiguous in memory, which is
	// beneficial for the inner k-loop in the kernel.
	DEFINE_PROTO_STRUCT(b_layout, i_vec ^ j_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	auto m = C | get_length<'i'>();
	auto n = C | get_length<'j'>();

	// Initialize C and B: for all i, j
	traverser(C, B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		C[state] = (num_t)((i + j) % 100) / m;
		B[state] = (num_t)((n + i - j) % 100) / m;
	});

	// Initialize A as lower triangular, rest filled with -999
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);

		if (k <= i)
			A[state] = (num_t)((i + k) % 100) / m;
		else
			A[state] = (num_t)-999;
	});
}

// computation kernel
//
// Optimized SYMM kernel:
//   C := beta * C + alpha * A_symm * B
// where A_symm is the symmetric matrix defined by the stored lower triangle of A.
//
// Main optimizations against the original kernel:
//  - Pre-scale C by beta in a separate pass (still inside the SCoP).
//  - Hoist alpha * B[i, j] out of the inner k-loop (temp1).
//  - Keep the reduction for C[i, j] in a scalar accumulator (temp2).
//  - Store B in column-major to make B[k, j] along k unit-stride.
//  - Block the j-dimension (columns) with into_blocks_dynamic for better cache
//    locality; each j-tile is processed as a unit.
//
// The algebra is equivalent to the original kernel:
//   for each (i, j):
//     - for k < i:
//         C[k, j] += alpha * B[i, j] * A[i, k]
//         temp2    +=        B[k, j] * A[i, k]
//     - then
//         C[i, j] = beta * C[i, j]
//                   + alpha * B[i, j] * A[i, i]
//                   + alpha * temp2
//
// Here we move the beta-scaling to a separate pass and then only add the
// alpha*A*B contributions.
[[gnu::flatten, gnu::noinline]]
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// Tile size for columns (j dimension). This is a moderate default; for
	// real tuning this can be adapted to the target cache sizes.
	constexpr std::size_t tile_j = 64;

	#pragma scop
	// -------------------------------------------------------------------------
	// 1. Pre-scale C by beta
	//
	// After this pass, C contains beta * C0 and the subsequent symmetric update
	// only adds alpha * A_symm * B contributions.
	// -------------------------------------------------------------------------
	traverser(C).for_each([&](auto state) {
		C[state] *= beta;
	});

	// -------------------------------------------------------------------------
	// 2. Symmetric matrix-matrix multiply using only the stored lower-triangle A
	//
	// We apply a column blocking on the j-dimension via into_blocks_dynamic.
	// This introduces three dimensions:
	//   'J' : column-tile index
	//   'j' : intra-tile column index
	//   't' : guard dimension (unit length for valid lanes, 0 for padded lanes)
	//
	// Traverser states produced here still use logical indices 'i', 'j', 'k';
	// the extra 'J' and 't' are only used to guide the traversal and are
	// ignored by the C/A/B structures themselves.
	// -------------------------------------------------------------------------
	auto trav = traverser(C, A, B).order(
		into_blocks_dynamic<'j', 'J', 'j', 't'>(tile_j)
	);

	// Outer loop over column tiles
	trav.template for_dims<'J'>([&](auto trav_J) {
		// Loop over rows i. i remains sequential to preserve the symmetric
		// update pattern: each pair (i, k) with k < i is processed once.
		trav_J.template for_dims<'i'>([&](auto trav_Ji) {
			// For each (i, j) inside the current column tile:
			trav_Ji.template for_dims<'j', 't'>([&](auto trav_ij) {
				// trav_ij has fixed i, j (and t); k is still free.
				auto state_ij = trav_ij.state();

				auto i_idx = get_index<'i'>(state_ij);
				auto j_idx = get_index<'j'>(state_ij);

				// Hoist alpha * B[i, j] out of the k-loop:
				num_t bij   = B[state_ij];
				num_t temp1 = alpha * bij;
				num_t temp2 = (num_t)0;

				// Inner reduction over the strictly lower-triangular part: k < i.
				trav_ij.template for_dims<'k'>([&](auto trav_ijk) {
					auto state_ijk = trav_ijk.state();
					auto k_idx = get_index<'k'>(state_ijk);

					if (k_idx < i_idx) {
						// Symmetric pair (i, k) with k < i:
						//  - C[k, j] gets alpha * B[i, j] * A[i, k]
						//  - C[i, j] gets alpha * A[i, k] * B[k, j] (via temp2)
						auto s_kj = idx<'i', 'j'>(k_idx, j_idx);
						auto s_ik = idx<'i', 'k'>(i_idx, k_idx);

						num_t aik = A[s_ik];

						C[s_kj] += temp1 * aik;
						temp2   += B[s_kj] * aik;
					}
				});

				// Diagonal + accumulated lower-triangular contribution to C[i, j].
				// At this point C already contains beta * C0.
				num_t a_ii = A[idx<'i', 'k'>(i_idx, i_idx)];

				C[state_ij] += temp1 * a_ii + alpha * temp2;
			});
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

	// input data
	num_t alpha;
	num_t beta;

	// lengths for all dimensions
	auto set_lengths =
		noarr::set_length<'i'>(m)
		^ noarr::set_length<'j'>(n)
		^ noarr::set_length<'k'>(m);

	// data containers
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_symm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}