#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector> // for temporary column-wise recurrence buffers

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
	traverser(imgIn).for_each([&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		imgIn[state] = (num_t)(((313 * i + 991 * j) % 65536) / 65535.0f);
	});
}

// computation kernel
// imgIn, imgOut, y1, y2: i x j
//
// Optimizations wrt the original version:
//  - coefficient exponentials are precomputed once
//  - horizontal passes keep the original optimal “row-major” traversal
//  - vertical passes are transformed from column-major (strided) to
//    row-major (contiguous) traversal using per-column recurrence
//    buffers. This preserves semantics while greatly improving
//    cache locality and vectorization opportunities.
//  - reverse traversals use state arithmetic instead of reconstructing
//    full index states from scratch in the inner loop.
[[gnu::flatten, gnu::noinline]]
void kernel_deriche(num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) {
	using namespace noarr;

	num_t k;
	num_t a1, a2, a3, a4, a5, a6, a7, a8;
	num_t b1, b2, c1, c2;

#pragma scop
	// ------------------------------------------------------------------
	// Coefficient setup
	// ------------------------------------------------------------------
	const num_t one  = SCALAR_VAL(1.0);
	const num_t two  = SCALAR_VAL(2.0);
	const num_t zero = SCALAR_VAL(0.0);

	const num_t exp_neg_alpha  = EXP_FUN(-alpha);
	const num_t exp_neg_2alpha = EXP_FUN(SCALAR_VAL(-2.0) * alpha);
	const num_t exp_pos_2alpha = EXP_FUN(SCALAR_VAL(2.0) * alpha);

	k = (one - exp_neg_alpha) * (one - exp_neg_alpha) /
	    (one + two * alpha * exp_neg_alpha - exp_pos_2alpha);

	a1 = a5 = k;
	a2 = a6 = k * exp_neg_alpha * (alpha - one);
	a3 = a7 = k * exp_neg_alpha * (alpha + one);
	a4 = a8 = -k * exp_neg_2alpha;
	b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
	b2 = -exp_neg_2alpha;
	c1 = c2 = one;

	// lengths used for reverse traversals and temporary buffers
	const std::size_t w_len = imgIn | get_length<'i'>();
	const std::size_t h_len = imgIn | get_length<'j'>();

	// ------------------------------------------------------------------
	// 1st pass: causal filter along j, for each i (forward j)
	//
	// for (i=0; i<W; i++)
	//   ym1=0; ym2=0; xm1=0;
	//   for (j=0; j<H; j++)
	//     y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2; ...
	// ------------------------------------------------------------------
	traverser(y1, imgIn).template for_dims<'i'>([&](auto row_trav) {
		num_t ym1 = zero;
		num_t ym2 = zero;
		num_t xm1 = zero;

		// j is the remaining (contiguous) dimension
		row_trav.for_each([&](auto state) {
			const num_t x_cur = imgIn[state];
			const num_t y_cur = a1 * x_cur + a2 * xm1 + b1 * ym1 + b2 * ym2;

			y1[state] = y_cur;

			xm1 = x_cur;
			ym2 = ym1;
			ym1 = y_cur;
		});
	});

	// ------------------------------------------------------------------
	// 2nd pass: anti-causal filter along j, for each i (reverse j)
	//
	// Original C (per row i):
	//   yp1=0; yp2=0; xp1=0; xp2=0;
	//   for (j=H-1; j>=0; j--)
	//     y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2; ...
	//
	// Here we traverse j in 0..H-1 and map r -> j = H-1-r using
	// state arithmetic instead of reconstructing states.
	// This keeps memory accesses contiguous along j.
	// ------------------------------------------------------------------
	const std::size_t last_j = h_len - 1;

	traverser(y2, imgIn).template for_dims<'i'>([&](auto row_trav) {
		num_t yp1 = zero;
		num_t yp2 = zero;
		num_t xp1 = zero;
		num_t xp2 = zero;

		// base state for (i, last_j)
		auto base_state = row_trav.state() + idx<'j'>(last_j);

		row_trav.for_each([&](auto fwd_state) {
			const std::size_t r_j = get_index<'j'>(fwd_state);

			// rev_state = (i, last_j - r_j)
			auto rev_state = base_state - idx<'j'>(r_j);

			const num_t y_cur = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
			y2[rev_state] = y_cur;

			xp2 = xp1;
			xp1 = imgIn[rev_state];
			yp2 = yp1;
			yp1 = y_cur;
		});
	});

	// ------------------------------------------------------------------
	// Combine vertical passes (horizontal stage output)
	// imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
	// ------------------------------------------------------------------
	traverser(imgOut, y1, y2).for_each([&](auto state) {
		imgOut[state] = c1 * (y1[state] + y2[state]);
	});

	// ------------------------------------------------------------------
	// 3rd pass: causal filter along i, for each j
	//
	// Original C:
	//   for (j=0; j<H; j++)
	//     tm1=0; ym1=0; ym2=0;
	//     for (i=0; i<W; i++)
	//       y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2; ...
	//
	// Optimization:
	//   We interchange the loops to make the innermost traversal
	//   contiguous along j (row-major), but keep exact per-column
	//   recurrences by storing tm1,ym1,ym2 for each column j in
	//   separate temporary arrays:
	//
	//   tm1_col[j], ym1_col[j], ym2_col[j]
	//
	//   Then:
	//     tm1_col[j]=ym1_col[j]=ym2_col[j]=0 for all j
	//     for (i=0; i<W; i++)
	//       for (j=0; j<H; j++)
	//         ... use / update column-wise buffers ...
	//
	//   This is semantically equivalent but accesses imgOut/y1
	//   in a cache- and SIMD-friendly way.
	// ------------------------------------------------------------------
	std::vector<num_t> tm1_col(h_len, zero);
	std::vector<num_t> ym1_col(h_len, zero);
	std::vector<num_t> ym2_col(h_len, zero);

	traverser(y1, imgOut).template for_dims<'i'>([&](auto row_trav) {
		// Here i is fixed; for_each iterates j in 0..H-1
		row_trav.for_each([&](auto state) {
			const std::size_t j_idx = get_index<'j'>(state);

			num_t &tm1 = tm1_col[j_idx];
			num_t &ym1 = ym1_col[j_idx];
			num_t &ym2 = ym2_col[j_idx];

			const num_t x_cur = imgOut[state];
			const num_t y_cur = a5 * x_cur + a6 * tm1 + b1 * ym1 + b2 * ym2;

			y1[state] = y_cur;

			tm1 = x_cur;
			ym2 = ym1;
			ym1 = y_cur;
		});
	});

	// ------------------------------------------------------------------
	// 4th pass: anti-causal filter along i, for each j
	//
	// Original C:
	//   for (j=0; j<H; j++)
	//     tp1=0; tp2=0; yp1=0; yp2=0;
	//     for (i=W-1; i>=0; i--)
	//       y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2; ...
	//
	// Optimization (similar to 3rd pass):
	//   We again use per-column recurrence buffers but process
	//   rows in reverse order while iterating j contiguously.
	//   For i_r = 0..W-1 we map to real row i = W-1-i_r and use:
	//
	//     tp1_col[j], tp2_col[j], yp1_col[j], yp2_col[j]
	//
	//   This keeps memory access row-major while preserving
	//   the original anti-causal order along i.
	// ------------------------------------------------------------------
	std::vector<num_t> tp1_col(h_len, zero);
	std::vector<num_t> tp2_col(h_len, zero);
	std::vector<num_t> yp1_col(h_len, zero);
	std::vector<num_t> yp2_col(h_len, zero);

	const std::size_t last_i = w_len - 1;

	traverser(y2, imgOut).template for_dims<'i'>([&](auto row_trav_r) {
		// row_trav_r has i fixed to a "reverse index" i_r in 0..W-1
		auto r_state = row_trav_r.state();
		const std::size_t i_r = get_index<'i'>(r_state);
		const std::size_t i_idx = last_i - i_r; // real row index, descending

		// base state for row i_idx, j will be added later
		auto base_row_state = idx<'i'>(i_idx);

		row_trav_r.for_each([&](auto state_r) {
			const std::size_t j_idx = get_index<'j'>(state_r);

			// rev_state = (i_idx, j_idx)
			auto rev_state = base_row_state + idx<'j'>(j_idx);

			num_t &tp1 = tp1_col[j_idx];
			num_t &tp2 = tp2_col[j_idx];
			num_t &yp1 = yp1_col[j_idx];
			num_t &yp2 = yp2_col[j_idx];

			const num_t y_cur = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
			y2[rev_state] = y_cur;

			tp2 = tp1;
			tp1 = imgOut[rev_state];
			yp2 = yp1;
			yp1 = y_cur;
		});
	});

	// ------------------------------------------------------------------
	// Final combine
	// imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
	// ------------------------------------------------------------------
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