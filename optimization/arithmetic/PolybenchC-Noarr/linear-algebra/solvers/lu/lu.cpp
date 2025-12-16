#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "lu.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(order, noarr::hoist<'j'>());

	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A) {
	// A: i x j
	using namespace noarr;

	int n = A | get_length<'i'>();

	traverser(A) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i + 1) | [=](auto state) {
			A[state] = (num_t) (-(int)get_index<'j'>(state) % n) / n + 1;
		};

		inner ^ shift<'j'>(i + 1) | [=](auto state) {
			A[state] = 0;
		};

		A[inner.state() & idx<'j'>(i)] = 1;
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
template<typename Order = noarr::neutral_proto>
[[gnu::flatten, gnu::noinline]]
void kernel_lu(auto A, Order order = {}) {
	// A: i x j
	using namespace noarr;

	auto A_ik = A ^ rename<'j', 'k'>();
	auto A_kj = A ^ rename<'i', 'k'>();

	#pragma scop
	traverser(A, A_ik, A_kj) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i) | for_dims<'j'>([=](auto inner) {
			auto j = get_index<'j'>(inner);

			inner ^ span<'k'>(j) | [=](auto state) {
				A[state] -= A_ik[state] * A_kj[state];
			};

			A[inner] /= (A ^ fix<'i'>(j))[inner];
		});

		inner ^ shift<'j'>(i) ^ span<'k'>(i) ^  order | [=](auto state) {
			A[state] -= A_ik[state] * A_kj[state];
		};
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
	kernel_lu(A.get_ref(), tuning.order);

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
