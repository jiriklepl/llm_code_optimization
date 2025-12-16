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
	DEFINE_PROTO_STRUCT(order, noarr::hoist<'k'>());

	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: i x k
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	auto ni = C | get_length<'i'>();
	auto nk = A | get_length<'k'>();

	traverser(A, B) | [=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)((i * k + 1) % ni) / ni;
		B[state] = (num_t)((i * k + 2) % nk) / nk;
	};

	traverser(C) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 3) % ni) / nk;
	};
}

// computation kernel
template<class Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B, Order order = {}) {
	// C: i x j
	// A: i x k
	// B: i x k
	using namespace noarr;

	auto A_renamed = A ^ rename<'i', 'j'>();
	auto B_renamed = B ^ rename<'i', 'j'>();

	#pragma scop
	traverser(C, A, B, A_renamed, B_renamed) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i + 1) | for_dims<'j'>([=](auto inner) {
			C[inner] *= beta;
		});

		inner ^ span<'j'>(i + 1) ^ order | [=](auto state) {
			C[state] += A_renamed[state] * alpha * B[state] + B_renamed[state] * alpha * A[state];
		};
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nk = NK;

	// data
	num_t alpha;
	num_t beta;

	auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'k'>(nk) ^ noarr::set_length<'j'>(ni);

	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syr2k(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref(), tuning.order);

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
