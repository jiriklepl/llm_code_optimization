#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "2mm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();
constexpr auto l_vec = noarr::vector<'l'>();

struct tuning {
	DEFINE_PROTO_STRUCT(tmp_layout, j_vec ^ i_vec); // NI x NJ
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);   // NI x NK
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);   // NK x NJ
	DEFINE_PROTO_STRUCT(c_layout, l_vec ^ j_vec);   // NJ x NL
	DEFINE_PROTO_STRUCT(d_layout, l_vec ^ i_vec);   // NI x NL
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto C, auto D) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// A: i x k
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto ni = A | get_length<'i'>();
		A[state] = (num_t)((i * k + 1) % ni) / ni;
	});

	// B: k x j
	traverser(B).for_each([=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		auto nj = B | get_length<'j'>();
		B[state] = (num_t)(k * (j + 1) % nj) / nj;
	});

	// C: j x l
	traverser(C).for_each([=](auto state) {
		auto [j, l] = get_indices<'j', 'l'>(state);
		auto nl = C | get_length<'l'>();
		C[state] = (num_t)((j * (l + 3) + 1) % nl) / nl;
	});

	// D: i x l
	auto nk = A | get_length<'k'>(); // note: uses NK as in the original C init
	traverser(D).for_each([=](auto state) {
		auto [i, l] = get_indices<'i', 'l'>(state);
		D[state] = (num_t)(i * (l + 2) % nk) / nk;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
	using namespace noarr;

	#pragma scop
	// tmp[i][j] = sum_k alpha * A[i][k] * B[k][j]
	traverser(tmp, A, B).template for_dims<'i'>([=](auto t_i) {
		// for (j) tmp[i][j] = 0
		t_i.template for_each<'j'>([=](auto s_ij) {
			tmp[s_ij] = (num_t)0;
		});
		// for (k) for (j) tmp[i][j] += ...
		t_i.template for_dims<'k'>([=](auto t_ik) {
			t_ik.template for_each<'j'>([=](auto s_ikj) {
				tmp[s_ikj] += alpha * A[s_ikj] * B[s_ikj];
			});
		});
	});

	// D[i][l] = beta * D[i][l] + sum_j tmp[i][j] * C[j][l]
	traverser(D, tmp, C).template for_dims<'i'>([=](auto t_i) {
		// for (l) D[i][l] *= beta
		t_i.template for_each<'l'>([=](auto s_il) {
			D[s_il] *= beta;
		});
		// for (j) for (l) D[i][l] += tmp[i][j] * C[j][l]
		t_i.template for_dims<'j'>([=](auto t_ij) {
			t_ij.template for_each<'l'>([=](auto s_ijl) {
				D[s_ijl] += tmp[s_ijl] * C[s_ijl];
			});
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;

	// scalars
	num_t alpha;
	num_t beta;

	// set lengths for all dims
	auto set_lengths =
		noarr::set_length<'i'>(ni) ^
		noarr::set_length<'j'>(nj) ^
		noarr::set_length<'k'>(nk) ^
		noarr::set_length<'l'>(nl);

	// allocate bags
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B   = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);
	auto C   = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto D   = noarr::bag(noarr::scalar<num_t>() ^ tuning.d_layout ^ set_lengths);

	// initialize arrays
	init_array(alpha, beta, A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	// time kernel
	auto start = std::chrono::high_resolution_clock::now();

	kernel_2mm(alpha, beta, tmp.get_ref(), A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results to stderr
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, D.get_ref() ^ noarr::hoist<'i'>());
	}

	// print time to stdout
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}
