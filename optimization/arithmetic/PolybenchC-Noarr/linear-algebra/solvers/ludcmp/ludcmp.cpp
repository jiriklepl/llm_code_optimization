#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "ludcmp.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto A, auto b, auto x, auto y) {
	// A: i x j
	// b: i
	// x: i
	// y: i
	using namespace noarr;

	int n = A | get_length<'i'>();
	num_t fn = (num_t)n;

	traverser(b, x, y) | [=](auto state) {
		auto i = get_index<'i'>(state);

		x[state] = 0;
		y[state] = 0;
		b[state] = (i + 1) / fn / 2.0 + 4;
	};

	traverser(A) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i + 1) | [=](auto state) {
			int j = get_index<'j'>(state);

			A[state] = (num_t)(-j % n) / n + 1;
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
[[gnu::flatten, gnu::noinline]]
void kernel_ludcmp(auto A, auto b, auto x, auto y) {
	// A: i x j
	// b: i
	// x: i
	// y: i
	using namespace noarr;

	auto A_ik = A ^ rename<'j', 'k'>();
	auto A_kj = A ^ rename<'i', 'k'>();

	#pragma scop
	traverser(A, b, x, y, A_ik, A_kj) | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		inner ^ span<'j'>(i) | for_dims<'j'>([=](auto inner) {
			auto j = get_index<'j'>(inner);

			num_t w = A[inner];

			inner ^ span<'k'>(j) | [=, &w](auto state) {
				w -= A_ik[state] * A_kj[state];
			};

			A[inner] = w / (A ^ fix<'i'>(j))[inner];
		});

		inner ^ shift<'j'>(i) | for_dims<'j'>([=](auto inner) {
			num_t w = A[inner];

			inner ^ span<'k'>(i) | [=, &w](auto state) {
				w -= A_ik[state] * A_kj[state];
			};

			A[inner] = w;
		});
	});

	traverser(A, b, y) | for_dims<'i'>([=](auto inner) {
		num_t w = b[inner];

		inner ^ span<'j'>(get_index<'i'>(inner)) | for_each<'j'>([=, &w](auto state) {
			w -= A[state] * y[idx<'i'>(get_index<'j'>(state))];
		});

		y[inner] = w;
	});

	traverser(A, x) ^ reverse<'i'>() | for_dims<'i'>([=](auto inner) {
		auto i = get_index<'i'>(inner);

		num_t w = y[inner];

		inner ^ shift<'j'>(i + 1) | for_each<'j'>([=, &w](auto state) {
			w -= A[state] * x[idx<'i'>(get_index<'j'>(state))];
		});

		x[inner] = w / A[inner.state() & idx<'j'>(i)];
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
	auto b = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));
	auto x = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));
	auto y = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'i'>(n));

	// initialize data
	init_array(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_ludcmp(A.get_ref(), b.get_ref(), x.get_ref(), y.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, x);
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
