#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions (DATA_TYPE, DEFINE_PROTO_STRUCT, etc.)
#include "defines.hpp"

// include benchmark-specific definitions (N, ...)
#include "gemver.hpp"

using num_t = DATA_TYPE;

namespace {

// convenient dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

// layout tuning (separated from sizes)
struct tuning {
	// A is N x N, row-major: inner dim 'j', outer dim 'i'
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);

	// 1D vectors indexed primarily by 'i'
	DEFINE_PROTO_STRUCT(u_layout, i_vec);

	// 1D vectors indexed primarily by 'j'
	DEFINE_PROTO_STRUCT(v_layout, j_vec);
} tuning;

// initialization function
// A: i x j
// u1,u2,w,x,z: i
// v1,v2,y: j
void init_array(num_t &alpha, num_t &beta,
                auto A, auto u1, auto v1, auto u2, auto v2,
                auto w, auto x, auto y, auto z) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	// problem size
	std::size_t n_size = A | get_length<'i'>();
	num_t fn = (num_t)n_size;

	// initialize vectors indexed by 'i': u1, u2, x, w, z
	traverser(u1, u2, x, w, z).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		u1[state] = (num_t)i;
		u2[state] = (((num_t)i + (num_t)1) / fn) / (num_t)2.0;
		x[state]  = (num_t)0.0;
		w[state]  = (num_t)0.0;
		z[state]  = (((num_t)i + (num_t)1) / fn) / (num_t)9.0;
	});

	// initialize vectors indexed by 'j': v1, v2, y
	traverser(v1, v2, y).for_each([=](auto state) {
		auto j = get_index<'j'>(state);

		v1[state] = (((num_t)j + (num_t)1) / fn) / (num_t)4.0;
		v2[state] = (((num_t)j + (num_t)1) / fn) / (num_t)6.0;
		y[state]  = (((num_t)j + (num_t)1) / fn) / (num_t)8.0;
	});

	// initialize matrix A: A[i][j] = (i*j % n) / n
	traverser(A).template for_dims<'i'>([=](auto trav_i) {
		trav_i.template for_each<'j'>([=](auto state) {
			auto [i, j] = get_indices<'i', 'j'>(state);
			A[state] = (num_t)((i * j) % n_size) / fn;
		});
	});
}

// computation kernel
// A: i x j
// u1,u2,w,x,z: i
// v1,v2,y: j
[[gnu::flatten, gnu::noinline]]
void kernel_gemver(num_t alpha, num_t beta,
                   auto A, auto u1, auto v1, auto u2, auto v2,
                   auto w, auto x, auto y, auto z) {
	using namespace noarr;

	// Tile size for the column ('j') dimension. This is a tuning knob; 64
	// works well on typical x64 CPUs and keeps tiles friendly to caches
	// while being large enough for vectorization.
	constexpr auto tile_j = noarr::lit<64>;

	#pragma scop

	// ---------------------------------------------------------------------
	// 1) Rank-2 update:
	//    A[i,j] = A[i,j] + u1[i]*v1[j] + u2[i]*v2[j]
	//
	// We:
	//  - keep A row-major with 'j' as the fast dimension,
	//  - tile the 'j' dimension to improve cache locality of A and v*,
	//  - hoist u1[i] and u2[i] into scalars (registers) per row.
	// ---------------------------------------------------------------------
	traverser(A, u1, v1, u2, v2)
		// Split 'j' into blocks: major index 'J', intra-block index 'j',
		// and guard dimension 'r' to handle a tail if N is not divisible
		// by tile_j. The layout in memory is unchanged.
		.order(noarr::into_blocks_dynamic<'j', 'J', 'j', 'r'>(tile_j))
		// Outer loop over rows i
		.template for_dims<'i'>([=](auto trav_i) {
			// trav_i has a fixed row index 'i'
			auto s_i = trav_i.state();

			// Scalar replacement: load u1[i], u2[i] once per row and reuse
			// them across all j in that row.
			num_t u1_i = u1[s_i];
			num_t u2_i = u2[s_i];

			// Loop over column blocks J for this row
			trav_i.template for_dims<'J'>([=](auto trav_J) {
				// Iterate all valid (j,r) pairs inside this block.
				// The guard dimension 'r' has length 0 for out-of-range
				// elements in the last block and 1 elsewhere, so for_each
				// only visits in-range elements.
				trav_J.for_each([=](auto state) {
					// state contains indices i, j, J, r; bags ignore
					// indices they do not use.
					A[state] = A[state]
					         + u1_i * v1[state]
					         + u2_i * v2[state];
				});
			});
		});

	// ---------------------------------------------------------------------
	// 2) x := z + beta * A^T * y
	//
	// Original code:
	//   x[i] starts at 0,
	//   x[i] += beta * sum_j A[j,i] * y[j],
	//   x[i] += z[i].
	//
	// We rewrite this as:
	//   y_beta[j] = beta * y[j]
	//   x[i]      = z[i]
	//   x[i]     += sum_j A[j,i] * y_beta[j]
	//
	// This is mathematically equivalent:
	//   x[i] = z[i] + beta * sum_j A[j,i] * y[j].
	//
	// To get row-major access to A, we build a renamed view A_row such
	// that A_row[j,i] == A[i,j], traverse rows j and update all x[i].
	// ---------------------------------------------------------------------

	// Temporary: y_beta[j] = beta * y[j]; layout matches y.
	auto y_beta = noarr::bag(y.structure());

	traverser(y, y_beta).for_each([=](auto state) {
		y_beta[state] = beta * y[state];
	});

	// Fold "x += z" into initialization so that the GEMV adds only A^T*y.
	traverser(x, z).for_each([=](auto state) {
		x[state] = z[state];
	});

	// Row-major view: A_row[j,i] corresponds to A[i,j] in memory.
	// rename<'i','j','j','i'> swaps the logical dimension names while
	// keeping the underlying layout; new dims are 'j' (rows), 'i' (cols).
	auto A_row = A ^ noarr::rename<'i', 'j', 'j', 'i'>();

	// Traverse rows of A_row (i.e., rows of A) and accumulate into x.
	traverser(A_row, x, y_beta).template for_dims<'j'>([=](auto trav_j) {
		// trav_j has fixed 'j' (one row of A and one y_beta[j])
		auto s_j = trav_j.state();
		num_t yb = y_beta[s_j]; // == beta * y[j]

		// For this row j, walk all columns i, updating x[i].
		trav_j.for_each([=](auto state) {
			// state has 'j' and 'i'; x ignores 'j' and uses 'i'.
			x[state] = x[state] + A_row[state] * yb;
		});
	});

	// ---------------------------------------------------------------------
	// 3) w := w + alpha * A * x
	//
	// Original code:
	//   x is already updated, then
	//   w[i] = w[i] + alpha * sum_j A[i,j] * x[j]
	//
	// We pull alpha out of the O(N^2) GEMV by pre-scaling x:
	//   x_j[j]    = view of x indexed by 'j'  (x_j[j] == x[j])
	//   x_alpha[j] = alpha * x_j[j]
	//   w[i]      += sum_j A[i,j] * x_alpha[j]
	//
	// This reduces multiplications by alpha from O(N^2) to O(N).
	// We also tile the j-dimension (columns) similarly to step 1 and
	// accumulate w[i] in a scalar to reduce load/store traffic.
	// ---------------------------------------------------------------------

	// View of x indexed by 'j' so it can align with the 'j' dimension of A.
	auto x_j = x ^ noarr::rename<'i', 'j'>();

	// Temporary: x_alpha[j] = alpha * x_j[j]
	auto x_alpha = noarr::bag(x_j.structure());

	traverser(x_j, x_alpha).for_each([=](auto state) {
		x_alpha[state] = alpha * x_j[state];
	});

	// Final GEMV: w[i] += sum_j A[i,j] * x_alpha[j]
	traverser(A, w, x_alpha)
		.order(noarr::into_blocks_dynamic<'j', 'J', 'j', 'r'>(tile_j))
		.template for_dims<'i'>([=](auto trav_i) {
			// Fixed row i
			auto s_i = trav_i.state();

			// Accumulate w[i] in a scalar to reduce memory traffic
			num_t w_i = w[s_i];

			// Iterate column blocks J for this row
			trav_i.template for_dims<'J'>([&](auto trav_J) {
				// Within this block, traverse all valid (j,r) pairs
				trav_J.for_each([&](auto state) {
					// A[state] = A[i,j], x_alpha[state] = x_alpha[j]
					w_i += A[state] * x_alpha[state];
				});
			});

			// Write back the accumulated value
			w[s_i] = w_i;
		});

	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// scalar parameters
	num_t alpha;
	num_t beta;

	// set lengths for dimensions
	auto len_i = noarr::set_length<'i'>(n);
	auto len_j = noarr::set_length<'j'>(n);

	// allocate bags (owning containers)

	// A: N x N, dims 'i' (rows) x 'j' (cols), row-major
	auto A  = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ len_i ^ len_j);

	// u1, u2, w, x, z: length N, dim 'i'
	auto u1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ len_i);
	auto u2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ len_i);
	auto w  = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ len_i);
	auto x  = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ len_i);
	auto z  = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ len_i);

	// v1, v2, y: length N, dim 'j'
	auto v1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ len_j);
	auto v2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ len_j);
	auto y  = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ len_j);

	// initialize data
	init_array(alpha, beta,
	           A.get_ref(), u1.get_ref(), v1.get_ref(), u2.get_ref(), v2.get_ref(),
	           w.get_ref(), x.get_ref(), y.get_ref(), z.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemver(alpha, beta,
	              A.get_ref(), u1.get_ref(), v1.get_ref(), u2.get_ref(), v2.get_ref(),
	              w.get_ref(), x.get_ref(), y.get_ref(), z.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, w.get_ref());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}