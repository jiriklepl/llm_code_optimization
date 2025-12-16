#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "symm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// C: i x j
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k (M x M)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: k x j (name row as 'k' to access both B[i][j] and B[k][j])
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);
} tuning;

// initialization function
void init_array(int m, int n, num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// C[i][j] = ((i+j) % 100) / m
	traverser(C) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i + j) % 100) / m;
	};

	// B[row,col] with row named 'k' here: B[k][j] = ((n + k - j) % 100) / m
	traverser(B) | [=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)((n + k - j) % 100) / m;
	};

	// A lower-triangular: for k <= i: A[i][k] = ((i+k) % 100) / m; else -999
	traverser(A) | for_dims<'i'>([=](auto ti) {
		auto i = get_index<'i'>(ti.state());
		ti | for_each<'k'>([=](auto state) {
			auto k = get_index<'k'>(state);
			A[state] = (k <= i) ? (num_t)((i + k) % 100) / m : (num_t)-999;
		});
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_symm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// for i in 0..M-1
	//   for j in 0..N-1
	//     temp2 = 0
	//     for k in 0..i-1:
	//       C[k][j] += alpha * B[i][j] * A[i][k]
	//       temp2 += B[k][j] * A[i][k]
	//     C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2
	#pragma scop
	traverser(C, A, B) | for_dims<'i'>([=](auto ti) {
		ti | for_dims<'j'>([=](auto tij) {
			num_t temp2 = 0;

			auto ij = tij.state();
			auto i = get_index<'i'>(ij);

			// Precompute B[i][j]
			num_t Bij = B[ij & idx<'k'>(i)];

			// k-loop: k in [0, i)
			tij.order(noarr::slice<'k'>(0, i)) | for_each([=, &temp2](auto s) {
				// s has 'k', 'i', 'j'
				C[s] += alpha * Bij * A[s];
				temp2 += B[s] * A[s];
			});

			// diagonal update for C[i][j]
			auto ii = ij & idx<'k'>(i);
			C[ij] = beta * C[ij] + alpha * Bij * A[ii] + alpha * temp2;
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	int m = M;
	int n = N;

	// input data
	num_t alpha;
	num_t beta;

	auto set_lengths = noarr::set_length<'i'>(m) ^ noarr::set_length<'k'>(m) ^ noarr::set_length<'j'>(n);

	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(m, n, alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_symm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

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
