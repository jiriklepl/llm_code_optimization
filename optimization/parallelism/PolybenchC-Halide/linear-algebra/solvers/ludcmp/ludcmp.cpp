#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "ludcmp.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(v_layout, i_vec);
} tuning;

void init_array(auto A, auto b, auto x, auto y) {
	using namespace noarr;

	// n
	const std::size_t n = A | get_length<'i'>();

	// x = 0, y = 0, b = (i+1)/n/2 + 4
	traverser(b, x, y) | [=](auto s) {
		auto i = get_index<'i'>(s);
		x[s] = (num_t)0;
		y[s] = (num_t)0;
		b[s] = (num_t)((num_t)(i + 1) / (num_t)n / (num_t)2.0) + (num_t)4.0;
	};

	// build a unit lower-triangular-ish A
	traverser(A).template for_dims<'i'>([=](auto ti) {
		auto i = get_index<'i'>(ti.state());
		// j in [0, i]: A[i][j] = 1 - j/n
		ti.order(noarr::slice<'j'>(0, i + 1)).for_each([=](auto sij) {
			auto j = get_index<'j'>(sij);
			A[sij] = (num_t)1 - (num_t)j / (num_t)n;
		});
		// j in (i, n): A[i][j] = 0
		if (i + 1 < n)
			ti.order(noarr::slice<'j'>(i + 1, n - (i + 1))).for_each([=](auto sij) {
				A[sij] = (num_t)0;
			});
		// A[i][i] = 1
		A[ti.state() + noarr::idx<'j'>(i)] = (num_t)1;
	});

	// Make the matrix positive semi-definite: B = A * A^T; A = B
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	auto B_ref = B.get_ref();

	// B = 0
	noarr::traverser(B_ref) | [=](auto s) {
		// zero initialize
		B_ref[s] = (num_t)0;
	};

	// Prepare renamed views for accumulation over k
	auto A_rt = A ^ noarr::rename<'j', 'k'>();             // dims: i, k
	auto A_st = A ^ noarr::rename<'i', 'j', 'j', 'k'>();    // dims: j (former i), k

	// B[i][j] += A[i][k] * A[j][k]
	noarr::traverser(B_ref, A_rt, A_st) | [=](auto s) {
		B_ref[s] += A_rt[s] * A_st[s];
	};

	// A = B
	noarr::traverser(A, B_ref) | [=](auto s) {
		A[s] = B_ref[s];
	};
}

[[gnu::flatten, gnu::noinline]]
void kernel_ludcmp(auto A, auto b, auto x, auto y) {
	using namespace noarr;

	const std::size_t n = A | get_length<'i'>();

	#pragma scop
	// LU decomposition in-place
	traverser(A).template for_dims<'i'>([=](auto ti) {
		auto i = get_index<'i'>(ti.state());

		// for j in [0, i)
		if (i > 0) {
			ti.order(noarr::slice<'j'>(0, i)).for_each([=](auto sij) {
				num_t w = A[sij];
				auto j = get_index<'j'>(sij);

				if (j > 0) {
					// sum over k in [0, j): A[i][k] * A[k][j]
					auto A_ik = A ^ noarr::rename<'j', 'k'>() ^ noarr::fix<'i'>(i); // dims: k
					auto A_kj = A ^ noarr::rename<'i', 'k'>() ^ noarr::fix<'j'>(j); // dims: k
					noarr::traverser(A_ik, A_kj).order(noarr::slice<'k'>(0, j)).for_each([&](auto sk) {
						w -= A_ik[sk] * A_kj[sk];
					});
				}

				num_t Ajj = A[noarr::idx<'i', 'j'>(j, j)];
				A[sij] = w / Ajj;
			});
		}

		// for j in [i, n)
		ti.order(noarr::slice<'j'>(i, n - i)).for_each([=](auto sij) {
			num_t w = A[sij];

			if (i > 0) {
				auto j = get_index<'j'>(sij);
				auto A_ik = A ^ noarr::rename<'j', 'k'>() ^ noarr::fix<'i'>(i); // dims: k
				auto A_kj = A ^ noarr::rename<'i', 'k'>() ^ noarr::fix<'j'>(j); // dims: k
				noarr::traverser(A_ik, A_kj).order(noarr::slice<'k'>(0, i)).for_each([&](auto sk) {
					w -= A_ik[sk] * A_kj[sk];
				});
			}

			A[sij] = w;
		});
	});

	// Forward substitution: solve L*y = b
	traverser(y, b, A).template for_dims<'i'>([=](auto ti) {
		auto i = get_index<'i'>(ti.state());
		num_t w = b[ti.state()];

		if (i > 0) {
			auto yj  = y ^ noarr::rename<'i', 'j'>();         // dims: j
			auto Aij = A ^ noarr::fix<'i'>(i);                // dims: j
			noarr::traverser(Aij, yj).order(noarr::slice<'j'>(0, i)).for_each([&](auto sj) {
				w -= Aij[sj] * yj[sj];
			});
		}

		y[ti.state()] = w;
	});

	// Backward substitution: solve U*x = y
	traverser(x, y, A).template for_dims<'i'>([=](auto ti) {
		auto idx = get_index<'i'>(ti.state());
		auto idec = n - 1 - idx;

		num_t w = y[noarr::idx<'i'>(idec)];

		if (idec + 1 < n) {
			auto xj  = x ^ noarr::rename<'i', 'j'>();         // dims: j
			auto Aij = A ^ noarr::fix<'i'>(idec);             // dims: j
			noarr::traverser(xj, Aij).order(noarr::slice<'j'>(idec + 1, n - (idec + 1))).for_each([&](auto sj) {
				w -= Aij[sj] * xj[sj];
			});
		}

		num_t Aii = A[noarr::idx<'i', 'j'>(idec, idec)];
		x[noarr::idx<'i'>(idec)] = w / Aii;
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// allocate
	auto set_n = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_n);
	auto b = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ noarr::set_length<'i'>(n));
	auto x = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ noarr::set_length<'i'>(n));
	auto y = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ noarr::set_length<'i'>(n));

	// init
	init_array(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// kernel
	kernel_ludcmp(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print x
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
