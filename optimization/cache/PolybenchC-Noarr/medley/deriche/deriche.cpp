#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "deriche.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto w_vec = noarr::vector<'w'>();
constexpr auto h_vec = noarr::vector<'h'>();

struct tuning {
	DEFINE_PROTO_STRUCT(img_in_layout, h_vec ^ w_vec);
	DEFINE_PROTO_STRUCT(img_out_layout, h_vec ^ w_vec);

	DEFINE_PROTO_STRUCT(y1_layout, h_vec ^ w_vec);
	DEFINE_PROTO_STRUCT(y2_layout, h_vec ^ w_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, auto imgIn, auto) {
	// imgIn: w x h
	// imgOut: w x h
	using namespace noarr;

	alpha = (num_t)0.25;

	traverser(imgIn) | [=](auto state) {
		auto [w, h] = get_indices<'w', 'h'>(state);
		imgIn[state] = (num_t)((313 * w + 991 * h) % 65536) / 65535.0f;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_deriche(num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) {
	// imgIn: w x h
	// imgOut: w x h
	// y1: w x h
	// y2: w x h
	using namespace noarr;

	num_t k;
	num_t a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2;

	#pragma scop
	k = ((num_t)1.0 - std::exp(-alpha)) * ((num_t)1.0 - std::exp(-alpha)) / ((num_t)1.0 + (num_t)2.0 * alpha * std::exp(-alpha) - std::exp(((num_t)2.0 * alpha)));
	a1 = a5 = k;
	a2 = a6 = k * std::exp(-alpha) * (alpha - (num_t)1.0);
	a3 = a7 = k * std::exp(-alpha) * (alpha + (num_t)1.0);
	a4 = a8 = -k * std::exp(((num_t)(-2.0) * alpha));
	b1 = std::pow((num_t)2.0, -alpha);
	b2 = -std::exp(((num_t)(-2.0) * alpha));
	c1 = c2 = 1;

	traverser(imgIn, y1) | for_dims<'w'>([=](auto inner) {
		num_t ym1 = 0;
		num_t ym2 = 0;
		num_t xm1 = 0;

		inner | [&, y1, imgIn](auto state) {
			y1[state] = a1 * imgIn[state] + a2 * xm1 + b1 * ym1 + b2 * ym2;
			xm1 = imgIn[state];
			ym2 = ym1;
			ym1 = y1[state];
		};
	});

	traverser(imgIn, y2) | for_dims<'w'>([=](auto inner) {
		num_t yp1 = 0;
		num_t yp2 = 0;
		num_t xp1 = 0;
		num_t xp2 = 0;

		inner ^ reverse<'h'>() | [&, y2, imgIn](auto state) {
			y2[state] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
			xp2 = xp1;
			xp1 = imgIn[state];
			yp2 = yp1;
			yp1 = y2[state];
		};
	});


	traverser(y1, y2, imgOut) | [=](auto state) {
		imgOut[state] = c1 * (y1[state] + y2[state]);
	};

	traverser(imgOut, y1) | for_dims<'h'>([=](auto inner) {
		num_t tm1 = 0;
		num_t ym1 = 0;
		num_t ym2 = 0;

		inner | [&, y1, imgOut](auto state) {
			y1[state] = a5 * imgOut[state] + a6 * tm1 + b1 * ym1 + b2 * ym2;
			tm1 = imgOut[state];
			ym2 = ym1;
			ym1 = y1[state];
		};
	});

	traverser(imgOut, y2) | for_dims<'h'>([=](auto inner) {
		num_t tp1 = 0;
		num_t tp2 = 0;
		num_t yp1 = 0;
		num_t yp2 = 0;

		inner ^ reverse<'w'>() | [&, y2, imgOut](auto state) {
			y2[state] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
			tp2 = tp1;
			tp1 = imgOut[state];
			yp2 = yp1;
			yp1 = y2[state];
		};
	});

	traverser(y1, y2, imgOut) | [=](auto state) {
		imgOut[state] = c2 * (y1[state] + y2[state]);
	};
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nw = NW;
	std::size_t nh = NH;

	// data
	num_t alpha;
	auto imgIn = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_in_layout ^ noarr::set_length<'w'>(nw) ^ noarr::set_length<'h'>(nh));
	auto imgOut = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_out_layout ^ noarr::set_length<'w'>(nw) ^ noarr::set_length<'h'>(nh));

	auto y1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.y1_layout ^ noarr::set_length<'w'>(nw) ^ noarr::set_length<'h'>(nh));
	auto y2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.y2_layout ^ noarr::set_length<'w'>(nw) ^ noarr::set_length<'h'>(nh));

	// initialize data
	init_array(alpha, imgIn.get_ref(), imgOut.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_deriche(alpha, imgIn.get_ref(), imgOut.get_ref(), y1.get_ref(), y2.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, imgOut.get_ref() ^ noarr::hoist<'w'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
