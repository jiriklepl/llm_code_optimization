#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "seidel-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec); // C row-major: j inner, i outer
} tuning;

// initialization function
void init_array(auto A) {
	using namespace noarr;

	traverser(A) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (num_t(i) * (j + 2) + 2) / (A | get_length<'i'>());
	};
}

// computation kernel (Gauss–Seidel 2D)
[[gnu::flatten, gnu::noinline]]
void kernel_seidel_2d(int tsteps, auto A) {
	using namespace noarr;

	const std::size_t ni = A | get_length<'i'>();
	const std::size_t nj = A | get_length<'j'>();

	#pragma scop
	for (int t = 0; t < tsteps; t++) {
		// iterate interior: i = 1..N-2, j = 1..N-2 preserving i outside, j inside
		auto interior = noarr::slice<'i'>(1, ni - 2) ^ noarr::slice<'j'>(1, nj - 2);

		traverser(A).order(interior) | [=](auto s) {
			// s contains original indices (i, j), so neighbor offsets are in original coordinates
			auto s_im1_jm1 = noarr::neighbor<'i','j'>(s, -1, -1);
			auto s_im1_j   = noarr::neighbor<'i','j'>(s, -1,  0);
			auto s_im1_jp1 = noarr::neighbor<'i','j'>(s, -1, +1);

			auto s_i_jm1   = noarr::neighbor<'i','j'>(s,  0, -1);
			auto s_i_jp1   = noarr::neighbor<'i','j'>(s,  0, +1);

			auto s_ip1_jm1 = noarr::neighbor<'i','j'>(s, +1, -1);
			auto s_ip1_j   = noarr::neighbor<'i','j'>(s, +1,  0);
			auto s_ip1_jp1 = noarr::neighbor<'i','j'>(s, +1, +1);

			A[s] = (
				A[s_im1_jm1] + A[s_im1_j] + A[s_im1_jp1] +
				A[s_i_jm1]   + A[s]       + A[s_i_jp1]   +
				A[s_ip1_jm1] + A[s_ip1_j] + A[s_ip1_jp1]
			) / SCALAR_VAL(9.0);
		};
	}
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;
	using namespace noarr;

	// problem size
	std::size_t n = N;
	int tsteps = TSTEPS;

	// structure and bag
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_seidel_2d(tsteps, A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
