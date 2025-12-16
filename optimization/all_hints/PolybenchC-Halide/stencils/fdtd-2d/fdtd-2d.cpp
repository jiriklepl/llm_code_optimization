#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "fdtd-2d.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto t_vec = noarr::vector<'t'>();

struct tuning {
	DEFINE_PROTO_STRUCT(e_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(t_layout, t_vec);
} tuning;

// initialization
void init_array(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	traverser(fict) | [=](auto st) {
		auto t = get_index<'t'>(st);
		fict[st] = (num_t)t;
	};

	traverser(ex, ey, hz) | [=](auto s) {
		auto [i, j] = get_indices<'i', 'j'>(s);
		ex[s] = ((num_t)i * (j + 1)) / (ex | get_length<'i'>());
		ey[s] = ((num_t)i * (j + 2)) / (ey | get_length<'j'>());
		hz[s] = ((num_t)i * (j + 3)) / (hz | get_length<'i'>());
	};
}

// kernel
[[gnu::flatten, gnu::noinline]]
void kernel_fdtd_2d(auto ex, auto ey, auto hz, auto fict) {
	using namespace noarr;

	const std::size_t nx = ey | get_length<'i'>();
	const std::size_t ny = ey | get_length<'j'>();

	#pragma scop
	traverser(fict) | [=](auto st) {
		const num_t ft = fict[st];

		// ey[0][j] = _fict_[t]
		traverser(ey).order(noarr::fix<'i'>(0)) | for_each<'j'>([=](auto s) {
			ey[s] = ft;
		});

		// for i = 1..nx-1, for j = 0..ny-1:
		// ey[i][j] -= 0.5 * (hz[i][j] - hz[i-1][j])
		traverser(ey, hz).order(noarr::slice<'i'>(1, nx - 1)) | [=](auto s) {
			ey[s] = ey[s] - SCALAR_VAL(0.5) * (hz[s] - hz[noarr::neighbor<'i'>(s, -1)]);
		};

		// for i = 0..nx-1, for j = 1..ny-1:
		// ex[i][j] -= 0.5 * (hz[i][j] - hz[i][j-1])
		traverser(ex, hz).order(noarr::slice<'j'>(1, ny - 1)) | [=](auto s) {
			ex[s] = ex[s] - SCALAR_VAL(0.5) * (hz[s] - hz[noarr::neighbor<'j'>(s, -1)]);
		};

		// for i = 0..nx-2, for j = 0..ny-2:
		// hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
		traverser(hz, ex, ey)
			.order(noarr::slice<'i'>(0, nx - 1) ^ noarr::slice<'j'>(0, ny - 1))
			| [=](auto s) {
				auto sp_jp1 = s + noarr::idx<'j'>(1);
				auto sp_ip1 = s + noarr::idx<'i'>(1);
				hz[s] = hz[s] - SCALAR_VAL(0.7) * ((ex[sp_jp1] - ex[s]) + (ey[sp_ip1] - ey[s]));
			};
	};
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t tmax = TMAX;
	std::size_t nx = NX;
	std::size_t ny = NY;

	// set lengths proto
	auto set_lengths = noarr::set_length<'t'>(tmax) ^ noarr::set_length<'i'>(nx) ^ noarr::set_length<'j'>(ny);

	// bags
	auto ex = noarr::bag(noarr::scalar<num_t>() ^ tuning.e_layout ^ set_lengths);
	auto ey = noarr::bag(noarr::scalar<num_t>() ^ tuning.e_layout ^ set_lengths);
	auto hz = noarr::bag(noarr::scalar<num_t>() ^ tuning.e_layout ^ set_lengths);
	auto fict = noarr::bag(noarr::scalar<num_t>() ^ tuning.t_layout ^ set_lengths);

	// init
	init_array(ex.get_ref(), ey.get_ref(), hz.get_ref(), fict.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// kernel
	kernel_fdtd_2d(ex.get_ref(), ey.get_ref(), hz.get_ref(), fict.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, ex.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cout, ey.get_ref() ^ noarr::hoist<'i'>());
		noarr::serialize_data(std::cout, hz.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6) << duration.count() << std::endl;
}
