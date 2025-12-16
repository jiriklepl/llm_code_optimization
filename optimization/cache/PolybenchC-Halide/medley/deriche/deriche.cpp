#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "deriche.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(img_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, auto imgIn, auto imgOut) {
	using namespace noarr;

	alpha = (num_t)0.25;

	traverser(imgIn) | for_dims<'i'>([=](auto ti) {
		ti | for_each<'j'>([=](auto s) {
			auto [i, j] = get_indices<'i', 'j'>(s);
			imgIn[s] = (num_t)((313 * i + 991 * j) % 65536) / (num_t)65535.0;
		});
	});
	(void)imgOut; // imgOut is not initialized by the original C version
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_deriche(int w, int h, num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) {
	using namespace noarr;

	#pragma scop
	num_t k, a1, a2, a3, a4, a5, a6, a7, a8;
	num_t b1, b2, c1, c2;

	k  = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) * (SCALAR_VAL(1.0) - EXP_FUN(-alpha))
	   / (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) - EXP_FUN(SCALAR_VAL(2.0) * alpha));
	a1 = a5 = k;
	a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
	a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
	a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
	b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	c1 = c2 = (num_t)1;

	// for (i = 0; i < W; i++) { ym1=0; ym2=0; xm1=0; for (j = 0; j < H; j++) { ... } }
	traverser(imgIn, y1) | for_dims<'i'>([=](auto ti) {
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);
		num_t xm1 = SCALAR_VAL(0.0);

		ti | for_each<'j'>([=, &ym1, &ym2, &xm1](auto s) {
			y1[s] = a1 * imgIn[s] + a2 * xm1 + b1 * ym1 + b2 * ym2;
			xm1 = imgIn[s];
			ym2 = ym1;
			ym1 = y1[s];
		});
	});

	// for (i = 0; i < W; i++) { yp1=yp2=xp1=xp2=0; for (j = H-1; j >= 0; j--) { ... } }
	traverser(imgIn, y2) | for_dims<'i'>([=](auto ti) {
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);
		num_t xp1 = SCALAR_VAL(0.0);
		num_t xp2 = SCALAR_VAL(0.0);

		std::size_t Hlen = y2 | get_length<'j'>();
		ti | for_each<'j'>([=, &yp1, &yp2, &xp1, &xp2, Hlen](auto s_asc) {
			// map ascending j to descending j': j' = H-1 - j
			auto s = noarr::update_index<'j'>(s_asc, [=](auto j) { return Hlen - 1 - (std::size_t)j; });

			y2[s] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
			xp2 = xp1;
			xp1 = imgIn[s];
			yp2 = yp1;
			yp1 = y2[s];
		});
	});

	// for (i = 0; i < W; i++) for (j = 0; j < H; j++) imgOut = c1*(y1+y2)
	traverser(imgOut, y1, y2) | for_dims<'i'>([=](auto ti) {
		ti | for_each<'j'>([=](auto s) {
			imgOut[s] = c1 * (y1[s] + y2[s]);
		});
	});

	// for (j = 0; j < H; j++) { tm1=ym1=ym2=0; for (i = 0; i < W; i++) { ... } }
	traverser(imgOut, y1) | for_dims<'j'>([=](auto tj) {
		num_t tm1 = SCALAR_VAL(0.0);
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);

		tj | for_each<'i'>([=, &tm1, &ym1, &ym2](auto s) {
			y1[s] = a5 * imgOut[s] + a6 * tm1 + b1 * ym1 + b2 * ym2;
			tm1 = imgOut[s];
			ym2 = ym1;
			ym1 = y1[s];
		});
	});

	// for (j = 0; j < H; j++) { tp1=tp2=yp1=yp2=0; for (i = W-1; i >= 0; i--) { ... } }
	traverser(imgOut, y2) | for_dims<'j'>([=](auto tj) {
		num_t tp1 = SCALAR_VAL(0.0);
		num_t tp2 = SCALAR_VAL(0.0);
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);

		std::size_t Wlen = y2 | get_length<'i'>();
		tj | for_each<'i'>([=, &tp1, &tp2, &yp1, &yp2, Wlen](auto s_asc) {
			// map ascending i to descending i': i' = W-1 - i
			auto s = noarr::update_index<'i'>(s_asc, [=](auto i) { return Wlen - 1 - (std::size_t)i; });

			y2[s] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
			tp2 = tp1;
			tp1 = imgOut[s];
			yp2 = yp1;
			yp1 = y2[s];
		});
	});

	// for (i = 0; i < W; i++) for (j = 0; j < H; j++) imgOut = c2*(y1+y2)
	traverser(imgOut, y1, y2) | for_dims<'i'>([=](auto ti) {
		ti | for_each<'j'>([=](auto s) {
			imgOut[s] = c2 * (y1[s] + y2[s]);
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	int w = W;
	int h = H;

	num_t alpha;

	auto set_lengths = noarr::set_length<'i'>(w) ^ noarr::set_length<'j'>(h);

	auto imgIn  = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto imgOut = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto y1     = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);
	auto y2     = noarr::bag(noarr::scalar<num_t>() ^ tuning.img_layout ^ set_lengths);

	init_array(alpha, imgIn.get_ref(), imgOut.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	kernel_deriche(w, h, alpha, imgIn.get_ref(), imgOut.get_ref(), y1.get_ref(), y2.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, imgOut.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
