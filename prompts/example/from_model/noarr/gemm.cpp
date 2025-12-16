#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ j_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 1) % (C | get_length<'i'>())) / (C | get_length<'i'>());
	});

	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)(i * (k + 1) % (A | get_length<'k'>())) / (A | get_length<'k'>());
	});

	traverser(B).for_each([=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 2) % (B | get_length<'j'>())) / (B | get_length<'j'>());
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	constexpr std::size_t tile_i = 16;
	constexpr std::size_t tile_j = 16;
	constexpr std::size_t tile_k = 16;

	#pragma scop
	traverser(C, A, B).order(into_blocks_dynamic<'i', 'I', 'i', 's'>(lit<tile_i>) // 's' is a guard dimension for partial tiles
	                       ^ into_blocks_dynamic<'j', 'J', 'j', 't'>(lit<tile_j>) // 't' is a guard dimension for partial tiles
	                       ^ into_blocks_dynamic<'k', 'K', 'k', 'u'>(lit<tile_k>) // 'u' is a guard dimension for partial tiles
	).template for_dims<'J'>([=](auto inner) {
		inner.template for_dims<'I'>([=](auto inner) {
			inner.template for_dims<'i', 's', 'j', 't'>([=](auto state) {
				C[state] *= beta;
			});

			inner.template for_dims<'K', 'i', 's', 'k', 'u', 'j', 't'>([=](auto state) {
				C[state] += alpha * A[state] * B[state];
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

	// input data
	num_t alpha;
	num_t beta;

	auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'k'>(nk);

	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}
