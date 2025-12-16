#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "syrk.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// C: i x j (N x N)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k (N x M)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// A[i][k] = ((i*k+1) % n) / n
	{
		auto n_len = A | get_length<'i'>();
		traverser(A) | [=](auto state) {
			auto [i, k] = get_indices<'i', 'k'>(state);
			A[state] = (num_t)((i * k + 1) % n_len) / n_len;
		};
	}

	// C[i][j] = ((i*j+2) % m) / m
	{
		// m comes from the 'k' dimension of A
		auto m_len = A | get_length<'k'>();
		traverser(C) | [=](auto state) {
			auto [i, j] = get_indices<'i', 'j'>(state);
			C[state] = (num_t)((i * j + 2) % m_len) / m_len;
		};
	}
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
	using namespace noarr;

	// Form C := alpha*A*A^T + beta*C, updating only the lower triangle (j <= i)
	#pragma scop
	traverser(C, A) | for_dims<'i'>([=](auto ti) {
		auto si = ti.state();
		auto i = get_index<'i'>(si);

		// Scale C[i][j] for j in [0..i]
		ti.order(noarr::slice<'j'>(0, i + 1)).for_each([=](auto s) {
			C[s] *= beta;
		});

		// Accumulate over k, for j in [0..i]
		ti | for_dims<'k'>([=](auto tik) {
			tik.order(noarr::slice<'j'>(0, i + 1)).for_each([=](auto s) {
				// s has i, j, k
				auto j = get_index<'j'>(s);

				// A[i][k]
				num_t a_ik = A[s]; // 'j' is ignored by A

				// A[j][k] -> replace index 'i' in the state with 'j'
				auto s_j = noarr::make_state<noarr::index_in<'i'>>(j) & s;
				num_t a_jk = A[s_j];

				C[s] += alpha * a_ik * a_jk;
			});
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t m = M;

	// input data
	num_t alpha;
	num_t beta;

	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(m);

	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syrk(alpha, beta, C.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
