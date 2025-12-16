#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "syr2k.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // C: i x j
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // A: i x k (k == m)
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ i_vec); // B: i x k (k == m)
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// A[i][k] = ((i*k+1) % n) / n
	traverser(A) | [=](auto s) {
		auto [i, k] = get_indices<'i', 'k'>(s);
		std::size_t n_len = A | get_length<'i'>();
		A[s] = (num_t)((i * k + 1) % n_len) / (num_t)n_len;
	};

	// B[i][k] = ((i*k+2) % m) / m
	traverser(B) | [=](auto s) {
		auto [i, k] = get_indices<'i', 'k'>(s);
		std::size_t m_len = B | get_length<'k'>();
		B[s] = (num_t)((i * k + 2) % m_len) / (num_t)m_len;
	};

	// C[i][j] = ((i*j+3) % n) / m
	traverser(C) | [=](auto s) {
		auto [i, j] = get_indices<'i', 'j'>(s);
		std::size_t n_len = C | get_length<'i'>();
		std::size_t m_len = A | get_length<'k'>();
		C[s] = (num_t)((i * j + 3) % n_len) / (num_t)m_len;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	#pragma scop
	traverser(C, A, B) | for_dims<'i'>([=](auto t_i) {
		auto i_state = t_i.state();
		std::size_t i_val = get_index<'i'>(i_state);

		// for (j = 0; j <= i; j++) C[i][j] *= beta;
		t_i.order(noarr::slice<'j'>(0, i_val + 1)) | for_each<'j'>([=](auto s) {
			C[s] *= beta;
		});

		// for (k = 0; k < m; k++) for (j = 0; j <= i; j++) C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
		t_i | for_dims<'k'>([=](auto t_ik) {
			t_ik.order(noarr::slice<'j'>(0, i_val + 1)) | for_each<'j'>([=](auto s) {
				auto j_val = get_index<'j'>(s);
				auto k_val = get_index<'k'>(s);
				auto i_curr = get_index<'i'>(s);

				// A[j][k] and B[j][k]
				num_t Ajk = A[noarr::idx<'i', 'k'>(j_val, k_val)];
				num_t Bjk = B[noarr::idx<'i', 'k'>(j_val, k_val)];

				// A[i][k] and B[i][k] (can use s directly)
				num_t Aik = A[s];
				num_t Bik = B[s];

				C[s] += Ajk * alpha * Bik + Bjk * alpha * Aik;
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
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syr2k(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

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
