#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "fdtd-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(ex_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(ey_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(hz_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto ex, auto ey, auto hz, auto _fict_) {
	// ex: i x j
	// ey: i x j
	// hz: i x j
	// _fict_: t
	using namespace noarr;

	auto ni = ex | get_length<'i'>();
	auto nj = ex | get_length<'j'>();

	traverser(_fict_) | [=](auto state) {
		auto t = get_index<'t'>(state);
		_fict_[state] = t;
	};

	traverser(ex, ey, hz) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		ex[state] = ((num_t) i * (j + 1)) / ni;
		ey[state] = ((num_t) i * (j + 2)) / nj;
		hz[state] = ((num_t) i * (j + 3)) / ni;
	};
}


// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_fdtd_2d(auto ex, auto ey, auto hz, auto _fict_) {
	// ex: i x j
	// ey: i x j
	// hz: i x j
	// _fict_: t
	using namespace noarr;

	#pragma scop
	traverser(ex, ey, hz, _fict_) | for_dims<'t'>([=](auto inner) {
		inner ^ shift<'i'>(1) | for_each<'j'>([=](auto state) {
			ey[state & idx<'i'>(0)] = _fict_[state];
		});

		inner ^ shift<'i'>(1) | [=](auto state) {
			ey[state] = ey[state] - (num_t).5 * (hz[state] - hz[state - idx<'i'>(1)]);
		};

		inner ^ shift<'j'>(1) | [=](auto state) {
			ex[state] = ex[state] - (num_t).5 * (hz[state] - hz[state - idx<'j'>(1)]);
		};

		inner ^
			span<'i'>(0, (ex | get_length<'i'>()) - 1) ^
			span<'j'>(0, (ex | get_length<'j'>()) - 1) |
			[=](auto state) {
				hz[state] = hz[state] - (num_t).7 * (
					ex[state + idx<'j'>(1)] -
					ex[state] +
					ey[state + idx<'i'>(1)] -
					ey[state]);
			};
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t t = TMAX;
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto ex = noarr::bag(noarr::scalar<num_t>() ^ tuning.ex_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));
	auto ey = noarr::bag(noarr::scalar<num_t>() ^ tuning.ey_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));
	auto hz = noarr::bag(noarr::scalar<num_t>() ^ tuning.hz_layout ^ noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj));
	auto _fict_ = noarr::bag(noarr::scalar<num_t>() ^ noarr::vector<'t'>(t));

	// initialize data
	init_array(ex.get_ref(), ey.get_ref(), hz.get_ref(), _fict_.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_fdtd_2d(ex.get_ref(), ey.get_ref(), hz.get_ref(), _fict_.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, ex.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cout, ey.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cout, hz.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
