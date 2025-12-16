#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "gesummv.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);   // A: i x j
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);   // B: i x j
	DEFINE_PROTO_STRUCT(x_layout, j_vec);           // x: j
	DEFINE_PROTO_STRUCT(y_layout, i_vec);           // y: i
	DEFINE_PROTO_STRUCT(tmp_layout, i_vec);         // tmp: i
} tuning;

// initialization function
void init_array(int n, num_t &alpha, num_t &beta, auto A, auto B, auto X) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	traverser(X) | [=](auto state) {
		auto j = get_index<'j'>(state);
		X[state] = (num_t)(j % (X | get_length<'j'>())) / (X | get_length<'j'>());
	};

	traverser(A, B) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto nlen = (A | get_length<'i'>());
		A[state] = (num_t)((i * j + 1) % nlen) / nlen;
		B[state] = (num_t)((i * j + 2) % nlen) / nlen;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gesummv(num_t alpha, num_t beta, auto A, auto B, auto TMP, auto X, auto Y) {
	using namespace noarr;

	#pragma scop
	traverser(A, B, TMP, X, Y) | for_dims<'i'>([=](auto inner) {
		auto si = inner.state(); // contains fixed 'i'

		TMP[si] = (num_t)0.0;
		Y[si] = (num_t)0.0;

		inner | for_each<'j'>([=](auto s) {
			TMP[s] = A[s] * X[s] + TMP[s];
			Y[s] = B[s] * X[s] + Y[s];
		});

		Y[si] = alpha * TMP[si] + beta * Y[si];
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

	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout   ^ set_lengths);
	auto B   = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout   ^ set_lengths);
	auto TMP = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);
	auto X   = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout   ^ set_lengths);
	auto Y   = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout   ^ set_lengths);

	// initialize data
	init_array((int)n, alpha, beta, A.get_ref(), B.get_ref(), X.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gesummv(alpha, beta, A.get_ref(), B.get_ref(), TMP.get_ref(), X.get_ref(), Y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, Y.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
