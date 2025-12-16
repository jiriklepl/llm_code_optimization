#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "gramschmidt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(r_layout, j_vec ^ k_vec);
	DEFINE_PROTO_STRUCT(q_layout, k_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto R, auto Q) {
	// A: i x k
	// R: k x j
	// Q: i x k
	using namespace noarr;

	auto ni = A | get_length<'i'>();

	traverser(A, Q) | [=](auto state) {
		auto i = get_index<'i'>(state);
		auto k = get_index<'k'>(state);

		A[state] =(((num_t)((i * k) % ni) / ni) * 100) + 10;
		Q[state] = 0.0;
	};

	traverser(R) | [=](auto state) {
		R[state] = 0.0;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gramschmidt(auto A, auto R, auto Q) {
	// A: i x k
	// R: k x j
	// Q: i x k
	using namespace noarr;

	auto A_ij = A ^ rename<'k', 'j'>();

	#pragma scop
	traverser(A_ij, R, Q) | for_dims<'k'>([=](auto inner) {
		auto k = get_index<'k'>(inner);

		num_t norm = 0;

		inner | for_each<'i'>([=, &norm](auto state) {
			norm += A[state] * A[state];
		});

		auto R_diag = R ^ fix<'j'>(k);

		R_diag[inner] = std::sqrt(norm);

		inner | for_each<'i'>([=](auto state) {
			Q[state] = A[state] / R_diag[state];
		});

		inner ^ shift<'j'>(k + 1) | for_dims<'j'>([=](auto inner) {
			R[inner] = 0;

			inner | [=](auto state) {
				R[state] = R[state] + Q[state] * A_ij[state];
			};

			inner | [=](auto state) {
				A_ij[state] = A_ij[state] - Q[state] * R[state];
			};
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

	// data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'k'>(nj));
	auto R = noarr::bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ noarr::set_length<'k'>(nj) ^ noarr::set_length<'j'>(nj));
	auto Q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'k'>(nj));

	// initialize data
	init_array(A.get_ref(), R.get_ref(), Q.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gramschmidt(A.get_ref(), R.get_ref(), Q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, R.get_ref() ^ noarr::hoist<'k'>());
		noarr::serialize_data(std::cout, Q.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
