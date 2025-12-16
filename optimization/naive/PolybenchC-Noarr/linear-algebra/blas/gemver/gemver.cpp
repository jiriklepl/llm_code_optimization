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
//
// The implementation below keeps the mathematical computation identical to
// the original PolyBench GEMVER kernel, but restructures traversals to
// improve cache locality and reduce redundant memory traffic:
//
// 1) In the rank-1 update of A, we cache u1[i] and u2[i] per row.
// 2) The x-update that originally used A[j][i] is rewritten algebraically
//    as x[j] += beta * A[i][j] * y[i], which allows a row-major traversal
//    of A while using Noarr `rename` views for x and y. This keeps A and
//    x accesses contiguous in memory.
// 3) The w-update accumulates w[i] in a register and writes it back once
//    per row, avoiding repeated loads/stores of w[i] inside the inner loop.
//
// All loop nests are expressed using Noarr traversers as required.
[[gnu::flatten, gnu::noinline]]
void kernel_gemver(num_t alpha, num_t beta,
                   auto A, auto u1, auto v1, auto u2, auto v2,
                   auto w, auto x, auto y, auto z) {
	using namespace noarr;

	#pragma scop

	// ---------------------------------------------------------------------
	// 1) Rank-1 update:
	//    A[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j];
	//
	// Traverse A row-major: outer 'i', inner 'j'.
	// Cache u1[i] and u2[i] once per row to keep them in registers.
	// ---------------------------------------------------------------------
	traverser(A, u1, v1, u2, v2).template for_dims<'i'>([=](auto trav_i) {
		// `trav_i` has dimension 'i' fixed, 'j' remaining
		auto state_i = trav_i.state();
		const num_t u1_i = u1[state_i];
		const num_t u2_i = u2[state_i];

		trav_i.for_each([=](auto state) {
			// state contains both 'i' and 'j'
			A[state] = A[state]
				+ u1_i * v1[state]
				+ u2_i * v2[state];
		});
	});

	// ---------------------------------------------------------------------
	// 2) x[i] = x[i] + beta * A[j][i] * y[j];
	//
	// This is algebraically equivalent to:
	//    for all j: x[j] += sum_i beta * A[i][j] * y[i]
	//
	// We implement the latter form, which allows us to iterate A in its
	// natural row-major order (outer 'i', inner 'j').
	//
	// To keep the code expressive while preserving the original dimensions:
	//   - Create a view y_i with dim 'i' so that y_i[i] == y[i].
	//   - Create a view x_j with dim 'j' so that x_j[j] == x[j].
	//
	// Then:
	//   for each i:
	//     scaled_y = beta * y_i[i];
	//     for each j:
	//       x_j[j] += scaled_y * A[i][j];
	// ---------------------------------------------------------------------
	auto x_j = x ^ rename<'i', 'j'>(); // view: dim 'j' -> original x
	auto y_i = y ^ rename<'j', 'i'>(); // view: dim 'i' -> original y

	traverser(A, x_j, y_i).template for_dims<'i'>([=](auto trav_i) {
		auto state_i = trav_i.state();                 // contains index in 'i'
		const num_t scaled_y = beta * y_i[state_i];    // beta * y[i]

		trav_i.for_each([=](auto state) {
			// state has both 'i' and 'j':
			//   - A[state]  uses both 'i' and 'j'
			//   - x_j[state] uses only 'j' (ignores 'i')
			x_j[state] += scaled_y * A[state];
		});
	});

	// ---------------------------------------------------------------------
	// 3) x[i] = x[i] + z[i];
	//
	// Simple element-wise update over dim 'i'.
	// ---------------------------------------------------------------------
	traverser(x, z).for_each([=](auto state) {
		x[state] = x[state] + z[state];
	});

	// ---------------------------------------------------------------------
	// 4) w[i] = w[i] + alpha * A[i][j] * x[j];
	//
	// Reuse the x_j view from step (2): x_j[j] == x[j].
	// We accumulate w[i] in a scalar `wi` per row and write it back
	// once, reducing memory traffic on w.
	// ---------------------------------------------------------------------
	traverser(A, w, x_j).template for_dims<'i'>([=](auto trav_i) {
		auto state_i = trav_i.state(); // fixed 'i' for this row

		// Start from current w[i] and accumulate contributions in a register.
		num_t wi = w[state_i];

		trav_i.for_each([=, &wi](auto state) {
			// state has 'i' and 'j'; x_j[state] uses only 'j'.
			wi += alpha * A[state] * x_j[state];
		});

		// Write the final accumulated value back to w[i].
		w[state_i] = wi;
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