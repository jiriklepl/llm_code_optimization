#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "deriche.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// img[i][j] with j the innermost (contiguous) dimension, as in C (img[W][H])
	DEFINE_PROTO_STRUCT(img_layout, j_vec ^ i_vec);
} tuning;

// initialization function
// imgIn, imgOut: i x j
void init_array(num_t &alpha, auto imgIn, auto imgOut) {
	using namespace noarr;
	(void)imgOut; // not used, kept for interface similarity

	alpha = SCALAR_VAL(0.25); // parameter of the filter

	// input should be between 0 and 1 (grayscale image pixel)
	traverser(imgIn).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		imgIn[state] = (num_t)(((313 * i + 991 * j) % 65536) / 65535.0f);
	});
}

// computation kernel
// imgIn, imgOut, y1, y2: i x j
[[gnu::flatten, gnu::noinline]]
void kernel_deriche(num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) {
	using namespace noarr;

	num_t k;
	num_t a1, a2, a3, a4, a5, a6, a7, a8;
	num_t b1, b2, c1, c2;

#pragma scop
	// coefficient setup (unchanged from C version)
	k = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) *
	    (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) /
	    (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) - EXP_FUN(SCALAR_VAL(2.0) * alpha));
	a1 = a5 = k;
	a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
	a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
	a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
	b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	c1 = c2 = SCALAR_VAL(1.0);

	// lengths used for reverse traversals
	const std::size_t w_len = imgIn | get_length<'i'>();
	const std::size_t h_len = imgIn | get_length<'j'>();

	// 1st pass: causal filter along j, for each i (forward j)
	// for (i=0; i<W; i++)
	//   ym1=0; ym2=0; xm1=0;
	//   for (j=0; j<H; j++)
	//     y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2; ...
	traverser(y1, imgIn).template for_dims<'i'>([&](auto inner) {
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);
		num_t xm1 = SCALAR_VAL(0.0);

		inner.template for_each<'j'>([&](auto state) {
			y1[state] = a1 * imgIn[state] + a2 * xm1 + b1 * ym1 + b2 * ym2;
			xm1 = imgIn[state];
			ym2 = ym1;
			ym1 = y1[state];
		});
	});

	// 2nd pass: anti-causal filter along j, for each i (reverse j)
	// Original C: j from H-1 down to 0
	// Here we traverse j in 0..H-1 and map r -> j = H-1-r
	traverser(y2, imgIn).template for_dims<'i'>([&](auto inner) {
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);
		num_t xp1 = SCALAR_VAL(0.0);
		num_t xp2 = SCALAR_VAL(0.0);

		inner.template for_each<'j'>([&](auto state) {
			auto [i_idx, r_j] = get_indices<'i', 'j'>(state);
			std::size_t j_idx = h_len - 1 - r_j;
			auto rev_state = idx<'i', 'j'>(i_idx, j_idx);

			y2[rev_state] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
			xp2 = xp1;
			xp1 = imgIn[rev_state];
			yp2 = yp1;
			yp1 = y2[rev_state];
		});
	});

	// Combine vertical passes
	// imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
	traverser(imgOut, y1, y2).for_each([&](auto state) {
		imgOut[state] = c1 * (y1[state] + y2[state]);
	});

	// 3rd pass: causal filter along i, for each j (forward i)
	// for (j=0; j<H; j++)
	//   tm1=0; ym1=0; ym2=0;
	//   for (i=0; i<W; i++)
	//     y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2; ...
	traverser(y1, imgOut).template for_dims<'j'>([&](auto inner) {
		num_t tm1 = SCALAR_VAL(0.0);
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);

		inner.template for_each<'i'>([&](auto state) {
			y1[state] = a5 * imgOut[state] + a6 * tm1 + b1 * ym1 + b2 * ym2;
			tm1 = imgOut[state];
			ym2 = ym1;
			ym1 = y1[state];
		});
	});

	// 4th pass: anti-causal filter along i, for each j (reverse i)
	// Original C: i from W-1 down to 0
	// Here we traverse i in 0..W-1 and map r -> i = W-1-r
	traverser(y2, imgOut).template for_dims<'j'>([&](auto inner) {
		num_t tp1 = SCALAR_VAL(0.0);
		num_t tp2 = SCALAR_VAL(0.0);
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);

		inner.template for_each<'i'>([&](auto state) {
			auto [r_i, j_idx] = get_indices<'i', 'j'>(state);
			std::size_t i_idx = w_len - 1 - r_i;
			auto rev_state = idx<'i', 'j'>(i_idx, j_idx);

			y2[rev_state] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
			tp2 = tp1;
			tp1 = imgOut[rev_state];
			yp2 = yp1;
			yp1 = y2[rev_state];
		});
	});

	// Final combine
	// imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
	traverser(imgOut, y1, y2).for_each([&](auto state) {
		imgOut[state] = c2 * (y1[state] + y2[state]);
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t w = W;
	std::size_t h = H;

	// parameter
	num_t alpha;

	// lengths shared by all images
	auto set_lengths =
		noarr::set_length<'i'>(w) ^
		noarr::set_length<'j'>(h);

	// images: i x j
	auto imgIn  = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto imgOut = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto y1     = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto y2     = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);

	// initialize data
	init_array(alpha, imgIn.get_ref(), imgOut.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_deriche(alpha, imgIn.get_ref(), imgOut.get_ref(), y1.get_ref(), y2.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (acts as DCE prevention)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, imgOut.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}