#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gemver.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A[i][j] - row-major: j is inner, i is outer
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	// vectors by their natural loop dimensions
	DEFINE_PROTO_STRUCT(u_i_layout, i_vec);
	DEFINE_PROTO_STRUCT(v_j_layout, j_vec);
	DEFINE_PROTO_STRUCT(w_i_layout, i_vec);
	DEFINE_PROTO_STRUCT(x_i_layout, i_vec);
	DEFINE_PROTO_STRUCT(y_j_layout, j_vec);
	DEFINE_PROTO_STRUCT(z_i_layout, i_vec);
} tuning;

// initialization function
static void init_array(num_t &alpha, num_t &beta,
                       auto A, auto u1, auto v1, auto u2, auto v2,
                       auto w, auto x, auto y, auto z) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	std::size_t n = A | get_length<'i'>(); // square matrix
	num_t fn = (num_t)n;

	traverser(u1) | [=](auto si) {
		auto i = get_index<'i'>(si);
		u1[si] = (num_t)i;
	};
	traverser(u2) | [=](auto si) {
		auto i = get_index<'i'>(si);
		u2[si] = ((num_t)(i + 1) / fn) / (num_t)2.0;
	};
	traverser(v1) | [=](auto sj) {
		auto j = get_index<'j'>(sj);
		v1[sj] = ((num_t)(j + 1) / fn) / (num_t)4.0;
	};
	traverser(v2) | [=](auto sj) {
		auto j = get_index<'j'>(sj);
		v2[sj] = ((num_t)(j + 1) / fn) / (num_t)6.0;
	};
	traverser(y) | [=](auto sj) {
		auto j = get_index<'j'>(sj);
		y[sj] = ((num_t)(j + 1) / fn) / (num_t)8.0;
	};
	traverser(z) | [=](auto si) {
		auto i = get_index<'i'>(si);
		z[si] = ((num_t)(i + 1) / fn) / (num_t)9.0;
	};
	traverser(x) | [=](auto si) {
		x[si] = (num_t)0.0;
	};
	traverser(w) | [=](auto si) {
		w[si] = (num_t)0.0;
	};

	traverser(A) | [=](auto s) {
		auto [i, j] = get_indices<'i','j'>(s);
		A[s] = (num_t)((i * j) % n) / (num_t)n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
static void kernel_gemver(num_t alpha, num_t beta,
                          auto A, auto u1, auto v1, auto u2, auto v2,
                          auto w, auto x, auto y, auto z) {
	using namespace noarr;

	#pragma scop
	// A[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j]
	traverser(A, u1, v1, u2, v2) | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto s) {
			A[s] = A[s] + u1[s] * v1[s] + u2[s] * v2[s];
		});
	});

	// x[i] = x[i] + beta * A[j][i] * y[j]
	traverser(A, x, y) | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto s) {
			auto [i, j] = get_indices<'i','j'>(s);
			auto s_t = noarr::idx<'i','j'>(j, i); // transpose access
			x[s] = x[s] + beta * A[s_t] * y[s];
		});
	});

	// x[i] = x[i] + z[i]
	traverser(x, z) | [=](auto si) {
		x[si] = x[si] + z[si];
	};

	// w[i] = w[i] + alpha * A[i][j] * x[j]
	traverser(A, w, x) | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto s) {
			w[s] = w[s] + alpha * A[s] * x[s];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// input scalars
	num_t alpha;
	num_t beta;

	// common length setter
	auto set_n = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// bags
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_n);
	auto u1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_i_layout ^ set_n);
	auto v1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_j_layout ^ set_n);
	auto u2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_i_layout ^ set_n);
	auto v2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_j_layout ^ set_n);
	auto w  = noarr::bag(noarr::scalar<num_t>() ^ tuning.w_i_layout ^ set_n);
	auto x  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_i_layout ^ set_n);
	auto y  = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_j_layout ^ set_n);
	auto z  = noarr::bag(noarr::scalar<num_t>() ^ tuning.z_i_layout ^ set_n);

	// initialize arrays
	init_array(alpha, beta, A.get_ref(), u1.get_ref(), v1.get_ref(), u2.get_ref(), v2.get_ref(),
	           w.get_ref(), x.get_ref(), y.get_ref(), z.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemver(alpha, beta, A.get_ref(), u1.get_ref(), v1.get_ref(), u2.get_ref(), v2.get_ref(),
	              w.get_ref(), x.get_ref(), y.get_ref(), z.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// output
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, w.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
