#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "heat-3d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(layout3d, k_vec ^ j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto B) {
	using namespace noarr;

	const auto n = A | get_length<'i'>();

	traverser(A, B).for_each([=](auto state) {
		auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);
		num_t v = (num_t)(i + j + (n - k)) * SCALAR_VAL(10.0) / (num_t)n;
		A[state] = v;
		B[state] = v;
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_heat_3d(int tsteps, auto A, auto B) {
	using namespace noarr;

	const auto ni = A | get_length<'i'>();
	const auto nj = A | get_length<'j'>();
	const auto nk = A | get_length<'k'>();

	auto core = noarr::slice<'i'>(1, ni - 2) ^ noarr::slice<'j'>(1, nj - 2) ^ noarr::slice<'k'>(1, nk - 2);

	#pragma scop
	for(int t = 1; t <= tsteps; t++) {
		traverser(A, B).order(core).template for_dims<'i'>([=](auto trav_i) {
			trav_i.template for_dims<'j'>([=](auto trav_j) {
				trav_j.for_each([=](auto s) {
					auto a_c  = A[s];
					auto a_ip = A[s + noarr::idx<'i'>(1)];
					auto a_im = A[s - noarr::idx<'i'>(1)];
					auto a_jp = A[s + noarr::idx<'j'>(1)];
					auto a_jm = A[s - noarr::idx<'j'>(1)];
					auto a_kp = A[s + noarr::idx<'k'>(1)];
					auto a_km = A[s - noarr::idx<'k'>(1)];

					B[s] = SCALAR_VAL(0.125) * (a_ip - SCALAR_VAL(2.0) * a_c + a_im)
						+ SCALAR_VAL(0.125) * (a_jp - SCALAR_VAL(2.0) * a_c + a_jm)
						+ SCALAR_VAL(0.125) * (a_kp - SCALAR_VAL(2.0) * a_c + a_km)
						+ a_c;
				});
			});
		});

		traverser(A, B).order(core).template for_dims<'i'>([=](auto trav_i) {
			trav_i.template for_dims<'j'>([=](auto trav_j) {
				trav_j.for_each([=](auto s) {
					auto b_c  = B[s];
					auto b_ip = B[s + noarr::idx<'i'>(1)];
					auto b_im = B[s - noarr::idx<'i'>(1)];
					auto b_jp = B[s + noarr::idx<'j'>(1)];
					auto b_jm = B[s - noarr::idx<'j'>(1)];
					auto b_kp = B[s + noarr::idx<'k'>(1)];
					auto b_km = B[s - noarr::idx<'k'>(1)];

					A[s] = SCALAR_VAL(0.125) * (b_ip - SCALAR_VAL(2.0) * b_c + b_im)
						+ SCALAR_VAL(0.125) * (b_jp - SCALAR_VAL(2.0) * b_c + b_jm)
						+ SCALAR_VAL(0.125) * (b_kp - SCALAR_VAL(2.0) * b_c + b_km)
						+ b_c;
				});
			});
		});
	}
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t tsteps = TSTEPS;

	// allocate data
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout3d ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout3d ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_heat_3d((int)tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}
