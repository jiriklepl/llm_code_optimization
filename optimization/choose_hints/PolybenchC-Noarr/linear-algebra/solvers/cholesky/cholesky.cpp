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
void init_array(auto A) {
	// A: i x j
	using namespace noarr;

	int n = A | get_length<'i'>();

	traverser(A) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		auto A_ii = A ^ fix<'j'>(i);

		inner ^ span<'j'>(i + 1) | [=](auto state) {
			A[state] = (num_t) (-(int)get_index<'j'>(state) % n) / n + 1;
		};

		inner ^ shift<'j'>(i + 1) | [=](auto state) {
			A[state] = 0;
		};

		A_ii[inner] = 1;
	});

	// make A positive semi-definite
	auto B = bag(A.structure());
	auto B_ref = B.get_ref();

	auto A_ik = A ^ rename<'j', 'k'>();
	auto A_jk = A ^ rename<'i', 'j', 'j', 'k'>();

	traverser(B_ref) | [=](auto state) {
		B_ref[state] = 0;
	};

	traverser(B_ref, A_ik, A_jk) | [=](auto state) {
		B_ref[state] += A_ik[state] * A_jk[state];
	};

	traverser(A, B_ref) | [=](auto state) {
		A[state] = B_ref[state];
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_cholesky(auto A) {
	// A: i x j
	using namespace noarr;

	auto A_ik = A ^ rename<'j', 'k'>();
	auto A_jk = A ^ rename<'i', 'j', 'j', 'k'>();

	#pragma scop
	traverser(A, A_ik, A_jk) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i) | for_dims<'j'>([=](auto inner) {
			auto j = get_index<'j'>(inner);

			inner ^ span<'k'>(j) | [=](auto state) {
				A[state] -= A_ik[state] * A_jk[state];
			};

			A[inner] /= (A ^ fix<'i'>(j))[inner];
		});

		auto A_ii = A ^ fix<'j'>(i);

		inner ^ span<'k'>(i) | for_each<'k'>([=](auto state) {
			A_ii[state] -= A_ik[state] * A_ik[state];
		});

		A_ii[inner] = std::sqrt(A_ii[inner]);
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_cholesky(A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) [A = A.get_ref()] {
		std::cout << std::fixed << std::setprecision(2);
		noarr::traverser(A) | noarr::for_dims<'i'>([=](auto inner) {
			inner ^ noarr::span<'j'>(noarr::get_index<'i'>(inner) + 1) |
				[=](auto state) {
					std::cout << A[state] << " ";
				};

			std::cout << std::endl;
		});
	}();

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
