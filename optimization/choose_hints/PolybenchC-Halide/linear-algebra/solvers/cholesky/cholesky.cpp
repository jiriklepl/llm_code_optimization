#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "cholesky.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(int n, auto A) {
	using namespace noarr;

	// Fill A: lower triangle set to (-j % n)/n + 1, upper to 0, diag to 1
	traverser(A) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		if (j <= i) {
			int neg_mod = (-static_cast<int>(j)) % n;
			A[state] = static_cast<num_t>(neg_mod) / static_cast<num_t>(n) + static_cast<num_t>(1);
		} else {
			A[state] = static_cast<num_t>(0);
		}
		if (i == j) A[state] = static_cast<num_t>(1);
	};

	// Make the matrix positive semi-definite: B = A * A^T
	auto B = noarr::bag(A);

	// Views with renamed dims so we can traverse r,s,t explicitly
	auto B_rs = B.get_ref() ^ noarr::rename<'i', 'r', 'j', 's'>();
	auto A_rt = A ^ noarr::rename<'i', 'r', 'j', 't'>();
	auto A_st = A ^ noarr::rename<'i', 's', 'j', 't'>();
	auto A_rs = A ^ noarr::rename<'i', 'r', 'j', 's'>();

	// Zero B
	noarr::traverser(B_rs) | [=](auto state) {
		B_rs[state] = static_cast<num_t>(0);
	};

	// Accumulate B[r][s] += A[r][t] * A[s][t]
	noarr::traverser(B_rs, A_rt, A_st) | noarr::for_dims<'t'>([=](auto ttrav) {
		ttrav | noarr::for_dims<'r'>([=](auto rtrav) {
			rtrav | noarr::for_each<'s'>([=](auto state) {
				B_rs[state] += A_rt[state] * A_st[state];
			});
		});
	});

	// Copy back B -> A
	noarr::traverser(A_rs, B_rs) | [=](auto state) {
		A_rs[state] = B_rs[state];
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_cholesky(auto A) {
	using namespace noarr;

	#pragma scop
	noarr::traverser(A) | noarr::for_dims<'i'>([&](auto ti) {
		// j < i block
		ti | noarr::for_dims<'j'>([&](auto tij) {
			auto s_ij = tij.state();
			std::size_t i = noarr::get_index<'i'>(s_ij);
			std::size_t j = noarr::get_index<'j'>(s_ij);

			if (j < i) {
				// Views for k-iteration
				auto A_i_k = A ^ noarr::rename<'j', 'k'>();           // dims: i, k
				auto A_j_k = A ^ noarr::rename<'i', 'j', 'j', 'k'>(); // dims: j, k

				// for (k = 0; k < j; k++) A[i][j] -= A[i][k] * A[j][k];
				noarr::traverser(A_i_k, A_j_k) | noarr::for_dims<'k'>([&](auto tk) {
					auto s_ijk = s_ij & tk.state(); // contains i, j (fixed) and k
					std::size_t k = noarr::get_index<'k'>(s_ijk);
					if (k < j) {
						A[s_ij] -= A_i_k[s_ijk] * A_j_k[s_ijk];
					}
				});

				// A[i][j] /= A[j][j];
				num_t ajj = A[noarr::idx<'i', 'j'>(j, j)];
				A[s_ij] /= ajj;
			}
		});

		// i == j case:
		// for (k = 0; k < i; k++) A[i][i] -= A[i][k] * A[i][k];
		auto A_i_k = A ^ noarr::rename<'j', 'k'>(); // dims: i, k
		noarr::traverser(A_i_k) | noarr::for_dims<'k'>([&](auto tk) {
			auto s_ik = ti.state() & tk.state(); // i fixed by ti, k iterated
			std::size_t i = noarr::get_index<'i'>(s_ik);
			std::size_t k = noarr::get_index<'k'>(s_ik);
			if (k < i) {
				num_t aik = A_i_k[s_ik];
				A[noarr::idx<'i', 'j'>(i, i)] -= aik * aik;
			}
		});

		// A[i][i] = sqrt(A[i][i]);
		auto si = ti.state();
		std::size_t i = noarr::get_index<'i'>(si);
		A[noarr::idx<'i', 'j'>(i, i)] = std::sqrt(A[noarr::idx<'i', 'j'>(i, i)]);
	});
	#pragma endscop
}

// printing function
void print_array(int n, auto A) {
	using namespace noarr;

	std::cout << std::fixed << std::setprecision(2);

	noarr::traverser(A) | noarr::for_dims<'i'>([&](auto ti) {
		ti | noarr::for_dims<'j'>([&](auto tij) {
			auto s = tij.state();
			auto i = noarr::get_index<'i'>(s);
			auto j = noarr::get_index<'j'>(s);
			if (j <= i) {
				if (((i * static_cast<std::size_t>(n) + j) % 20) == 0) std::cout << "\n";
				std::cout << A[s] << ' ';
			}
		});
	});
	std::cout << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// allocate data
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array(static_cast<int>(n), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_cholesky(A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (lower triangle)
	if (argc > 0 && argv[0] != ""s) {
		print_array(static_cast<int>(n), A.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
