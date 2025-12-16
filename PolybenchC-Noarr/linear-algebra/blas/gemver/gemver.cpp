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

	#pragma scop

	// 1) A[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j];
	traverser(A, u1, v1, u2, v2).template for_dims<'i'>([=](auto trav_i) {
		trav_i.template for_each<'j'>([=](auto state) {
			A[state] = A[state]
				+ u1[state] * v1[state]
				+ u2[state] * v2[state];
		});
	});

	// 2) x[i] = x[i] + beta * A[j][i] * y[j];
	// create a transposed view of A so that A_T[i,j] == A[j,i]
	auto A_T = A ^ noarr::rename<'i', 'j', 'j', 'i'>();

	traverser(A_T, x, y).template for_dims<'i'>([=](auto trav_i) {
		trav_i.template for_each<'j'>([=](auto state) {
			x[state] = x[state] + beta * A_T[state] * y[state];
		});
	});

	// 3) x[i] = x[i] + z[i];
	traverser(x, z).for_each([=](auto state) {
		x[state] = x[state] + z[state];
	});

	// 4) w[i] = w[i] + alpha * A[i][j] * x[j];
	// create a view of x indexed by 'j' so that x_j[j] == x[j]
	auto x_j = x ^ noarr::rename<'i', 'j'>();

	traverser(A, w, x_j).template for_dims<'i'>([=](auto trav_i) {
		trav_i.template for_each<'j'>([=](auto state) {
			w[state] = w[state] + alpha * A[state] * x_j[state];
		});
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