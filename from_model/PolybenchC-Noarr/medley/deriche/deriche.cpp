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
	// ------------------------------------------------------------------
	// Coefficient setup (loop-invariant)
	// Precompute exponentials/powers once and reuse them.
	// ------------------------------------------------------------------
	const num_t exp_m_alpha  = EXP_FUN(-alpha);
	const num_t exp_p2_alpha = EXP_FUN(SCALAR_VAL(2.0) * alpha);
	const num_t exp_m2_alpha = EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	const num_t pow_2_m_alpha = POW_FUN(SCALAR_VAL(2.0), -alpha);

	k = (SCALAR_VAL(1.0) - exp_m_alpha) *
	    (SCALAR_VAL(1.0) - exp_m_alpha) /
	    (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * exp_m_alpha - exp_p2_alpha);

	a1 = a5 = k;
	a2 = a6 = k * exp_m_alpha * (alpha - SCALAR_VAL(1.0));
	a3 = a7 = k * exp_m_alpha * (alpha + SCALAR_VAL(1.0));
	a4 = a8 = -k * exp_m2_alpha;
	b1 = pow_2_m_alpha;
	b2 = -exp_m2_alpha;
	c1 = c2 = SCALAR_VAL(1.0);

	// lengths used for reverse traversals
	const std::size_t w_len = imgIn | get_length<'i'>();
	const std::size_t h_len = imgIn | get_length<'j'>();

	// ==================================================================
	// 1st pass: causal filter along j, for each i (forward j)
	//   y1[i][j] = a1*imgIn[i][j] + a2*imgIn[i][j-1] + b1*y1[i][j-1] + b2*y1[i][j-2]
	// This is a 1D IIR along the contiguous j-dimension, rows (i) independent.
	// ==================================================================
	traverser(y1, imgIn).template for_dims<'i'>([&](auto inner) {
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);
		num_t xm1 = SCALAR_VAL(0.0);

		inner.template for_each<'j'>([&](auto state) {
			// Load input once to help the compiler generate FMAs.
			const num_t x = imgIn[state];
			const num_t y_val = a1 * x + a2 * xm1 + b1 * ym1 + b2 * ym2;

			y1[state] = y_val;

			xm1 = x;
			ym2 = ym1;
			ym1 = y_val;
		});
	});

	// ==================================================================
	// 2nd pass: anti-causal filter along j, for each i (reverse j)
	// We traverse j in 0..H-1 and map r -> j = H-1-r, so that we still
	// use forward Noarr traversal but implement a backward recurrence.
	//
	// Original:
	//   yp1=yp2=xp1=xp2=0;
	//   for (j = H-1; j >= 0; --j) {
	//     y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
	//     xp2 = xp1; xp1 = imgIn[i][j];
	//     yp2 = yp1; yp1 = y2[i][j];
	//   }
	//
	// Optimization: fuse vertical combine
	//   imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
	// into this pass to avoid an extra full image sweep.
	// ==================================================================
	traverser(y2, imgIn, y1, imgOut).template for_dims<'i'>([&](auto inner) {
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);
		num_t xp1 = SCALAR_VAL(0.0);
		num_t xp2 = SCALAR_VAL(0.0);

		inner.template for_each<'j'>([&](auto state) {
			auto [i_idx, r_j] = get_indices<'i', 'j'>(state);
			const std::size_t j_idx = h_len - 1 - r_j; // true index: H-1, H-2, ..., 0
			auto rev_state = idx<'i', 'j'>(i_idx, j_idx);

			// Anti-causal output at (i, j_idx)
			const num_t y_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
			y2[rev_state] = y_val;

			// Fused vertical combine: produce intermediate imgOut directly.
			imgOut[rev_state] = c1 * (y1[rev_state] + y_val);

			// Update recurrence state using original input signal.
			xp2 = xp1;
			xp1 = imgIn[rev_state];
			yp2 = yp1;
			yp1 = y_val;
		});
	});

	// At this point, imgOut contains the vertically filtered image:
	//   imgOut = c1 * (y1_vert + y2_vert)

	// ==================================================================
	// 3rd pass: causal filter along i, for each j (forward i)
	//   y1[i][j] = a5*imgOut[i][j] + a6*imgOut[i-1][j]
	//              + b1*y1[i-1][j] + b2*y1[i-2][j]
	// Here, columns j are independent; recurrence is along i.
	// ==================================================================
	traverser(y1, imgOut).template for_dims<'j'>([&](auto inner) {
		num_t tm1 = SCALAR_VAL(0.0);
		num_t ym1 = SCALAR_VAL(0.0);
		num_t ym2 = SCALAR_VAL(0.0);

		inner.template for_each<'i'>([&](auto state) {
			const num_t x = imgOut[state];
			const num_t y_val = a5 * x + a6 * tm1 + b1 * ym1 + b2 * ym2;

			y1[state] = y_val;

			tm1 = x;
			ym2 = ym1;
			ym1 = y_val;
		});
	});

	// ==================================================================
	// 4th pass: anti-causal filter along i, for each j (reverse i)
	//
	// Original:
	//   tp1=tp2=yp1=yp2=0;
	//   for (i = W-1; i >= 0; --i) {
	//     y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
	//     tp2 = tp1; tp1 = imgOut[i][j];  // imgOut is from vertical combine
	//     yp2 = yp1; yp1 = y2[i][j];
	//   }
	//   // then final combine in a separate pass:
	//   imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
	//
	// Optimization: fuse the final combine into this pass. We must:
	//   - read the original imgOut[i][j] (vertical result) into 'orig'
	//     before overwriting imgOut with the final combined value,
	//   - use 'orig' for tp1 (the recurrence input signal).
	// ==================================================================
	traverser(y2, imgOut, y1).template for_dims<'j'>([&](auto inner) {
		num_t tp1 = SCALAR_VAL(0.0);
		num_t tp2 = SCALAR_VAL(0.0);
		num_t yp1 = SCALAR_VAL(0.0);
		num_t yp2 = SCALAR_VAL(0.0);

		inner.template for_each<'i'>([&](auto state) {
			auto [r_i, j_idx] = get_indices<'i', 'j'>(state);
			const std::size_t i_idx = w_len - 1 - r_i; // true index: W-1, W-2, ..., 0
			auto rev_state = idx<'i', 'j'>(i_idx, j_idx);

			// Read original vertically filtered input before overwriting.
			const num_t orig = imgOut[rev_state];

			// Anti-causal horizontal output at (i_idx, j_idx).
			const num_t y_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
			y2[rev_state] = y_val;

			// Fused final combine: write final imgOut in-place.
			imgOut[rev_state] = c2 * (y1[rev_state] + y_val);

			// Update recurrence state using the *original* vertical output.
			tp2 = tp1;
			tp1 = orig;
			yp2 = yp1;
			yp1 = y_val;
		});
	});

	// Note:
	// - We have eliminated two full-image passes:
	//     * vertical combine imgOut = c1*(y1 + y2)
	//     * final combine  imgOut = c2*(y1 + y2)
	//   by fusing them into the second vertical and fourth horizontal passes.
	// - All recurrences (vertical and horizontal, causal and anti-causal)
	//   are preserved exactly; only when and where we add the two passes
	//   together has changed, not what is computed.
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