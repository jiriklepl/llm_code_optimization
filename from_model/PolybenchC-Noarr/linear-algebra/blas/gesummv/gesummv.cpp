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
//
// Optimized GESUMMV:
//   y = alpha * A * x + beta * B * x
//
// Key optimizations relative to the straightforward version:
//   - Use per-row scalar accumulators (sumA, sumB) instead of repeatedly
//     loading/storing tmp[i] and y[i] inside the j-loop. This keeps the two
//     reductions in registers and writes tmp[i], y[i] exactly once per row.
//   - Tile both i (rows) and j (columns) with into_blocks_dynamic to improve
//     cache locality for A, B, and x without assuming that N is a multiple
//     of the tile sizes. The guard dimensions added by into_blocks_dynamic
//     are handled internally by the traverser when using order(...).
//   - Preserve the original reduction order over j for each i, so the
//     floating-point arithmetic is the same sequence of operations as in
//     the original kernel (modulo harmless reordering of memory accesses).
[[gnu::flatten, gnu::noinline]]
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto tmp, auto x, auto y) {
	using namespace noarr;

	// Tunable tile sizes; chosen to give reasonably large tiles while
	// still fitting well into caches on typical x64 CPUs.
	constexpr std::size_t tile_i = 64;   // rows per tile
	constexpr std::size_t tile_j = 256;  // columns per tile

	// Block both i and j using into_blocks_dynamic:
	//   - 'I' is the tile index for rows, 'i' the row inside a tile
	//   - 'J' is the tile index for columns, 'j' the column inside a tile
	//   - 'r' and 's' are guard dimensions (presence flags) that handle
	//     partial tiles when N is not divisible by tile_i / tile_j.
	//
	// We apply these only through traverser.order(...), so the states
	// passed into the lambdas still use the original indices 'i' and 'j';
	// the block indices ('I', 'J') and guards ('r', 's') remain internal
	// to the traverser and are not visible in the states we see.
	constexpr auto blocking =
		into_blocks_dynamic<'i', 'I', 'i', 'r'>(lit<tile_i>) ^
		into_blocks_dynamic<'j', 'J', 'j', 's'>(lit<tile_j>);

	#pragma scop
	traverser(A, B, tmp, x, y)
		// Reorder the internal traversal to work on tiles of rows and columns.
		.order(blocking)
		// Outer loop over row tiles I
		.template for_dims<'I'>([=](auto trav_I) {
			// Loop over rows i inside the current row tile I
			trav_I.template for_dims<'i'>([=](auto trav_i) {
				// trav_i identifies a single logical row i; the state
				// we obtain here uses the original dimension name 'i'
				// and can be used directly to index tmp and y.
				auto si = trav_i.state(); // contains index in 'i'

				// Per-row accumulators kept in registers
				num_t sumA = (num_t)0.0;
				num_t sumB = (num_t)0.0;

				// For this row i, iterate over all column tiles J...
				trav_i.template for_dims<'J'>([=, &sumA, &sumB](auto trav_J) {
					// ...and within each (i, J) tile, traverse all valid
					// columns j in that tile. The into_blocks_dynamic guards
					// ('r', 's') are handled internally by for_each: any
					// (I, i, J, j) combination that would be out of range
					// simply produces no state.
					trav_J.for_each([&](auto s) {
						// s has the original indices 'i' and 'j'
						auto j  = get_index<'j'>(s);
						auto sj = idx<'j'>(j); // state for x[j]

						num_t xj = x[sj];

						// Fused accumulation of A and B contributions using
						// the same x[j]. This is algebraically equivalent to:
						//   tmp[i] += A[i][j] * x[j];
						//   y[i]   += B[i][j] * x[j];
						sumA += A[s] * xj;
						sumB += B[s] * xj;
					});
				});

				// After finishing all j for this i:
				//   tmp[i] = Σ_j A[i,j]*x[j]
				//   y[i]   = alpha * tmp[i] + beta * Σ_j B[i,j]*x[j]
				// This exactly matches the original kernel's final result
				// for tmp[i] and y[i], but avoids O(N^2) loads/stores of
				// tmp[i] and y[i] inside the inner j-loop.
				tmp[si] = sumA;
				y[si]   = alpha * sumA + beta * sumB;
			});
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