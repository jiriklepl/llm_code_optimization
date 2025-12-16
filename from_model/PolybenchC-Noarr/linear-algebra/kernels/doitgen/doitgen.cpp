#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/structures_extended.hpp>
#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions (DATA_TYPE, NR, NQ, NP, ...)
#include "defines.hpp"

// include benchmark-specific definitions (NR, NQ, NP, etc.)
#include "doitgen.hpp"

using num_t = DATA_TYPE;

namespace {

// dimension prototypes
constexpr auto r_vec = noarr::vector<'r'>(); // corresponds to C index r (0..NR-1)
constexpr auto q_vec = noarr::vector<'q'>(); // corresponds to C index q (0..NQ-1)
constexpr auto p_vec = noarr::vector<'p'>(); // corresponds to C index p (0..NP-1)
constexpr auto s_vec = noarr::vector<'s'>(); // corresponds to C index s (0..NP-1)

// layouts (match C memory layout: last index varies fastest)
constexpr auto A_layout  = p_vec ^ q_vec ^ r_vec; // A[r][q][p]
constexpr auto C4_layout = p_vec ^ s_vec;         // C4[s][p]
constexpr auto sum_layout = p_vec;                // sum[p]

// initialization function
void init_array(auto A, auto C4) {
	using namespace noarr;

	// A[r][q][p] = ((r*q + p) % np) / np;
	traverser(A).for_each([=](auto state) {
		auto [r, q, p] = get_indices<'r', 'q', 'p'>(state);
		auto np_len = A | get_length<'p'>();

		A[state] = static_cast<num_t>(( (r * q + p) % np_len ))
			/ static_cast<num_t>(np_len);
	});

	// C4[s][p] = (s*p % np) / np;
	traverser(C4).for_each([=](auto state) {
		auto [s, p] = get_indices<'s', 'p'>(state);
		auto np_len = C4 | get_length<'p'>();

		C4[state] = static_cast<num_t>(( (s * p) % np_len ))
			/ static_cast<num_t>(np_len);
	});
}

// computation kernel
//
// Original mathematical kernel for each (r, q):
//
//   for (p)
//     sum[p] = 0;
//   for (s)
//     for (p)
//       sum[p] += A[r][q][s] * C4[s][p];
//   for (p)
//     A[r][q][p] = sum[p];
//
// Compared to the baseline Noarr implementation (p outer, s inner, loading
// A[r][q][s] in the innermost loop), this version:
//
//   * keeps the same per-element reduction order over s (only loop
//     interchange in (p,s)), so results are mathematically identical;
//   * iterates s outer and p inner, so we load A[r][q][s] once per s and
//     reuse it across all p;
//   * traverses C4[s][p] and sum[p] with unit stride along p, improving
//     cache behavior and making the inner loop more SIMD‑friendly.
//
// All loop nests are expressed using Noarr traversers only.
[[gnu::flatten, gnu::noinline]]
void kernel_doitgen(auto A, auto C4, auto sum) {
	using namespace noarr;

	// C reference kernel reminder:
	// for (r)
	//   for (q) {
	//     for (p) {
	//       sum[p] = 0;
	//       for (s)
	//         sum[p] += A[r][q][s] * C4[s][p];
	//     }
	//     for (p)
	//       A[r][q][p] = sum[p];
	//   }

#pragma scop
	// Outer loops over independent (r, q) slices of A.
	traverser(A).template for_dims<'r', 'q'>([&](auto trav_rq) {
		// Fixed indices for this slice
		auto state_rq = trav_rq.state();
		auto r_idx = get_index<'r'>(state_rq);
		auto q_idx = get_index<'q'>(state_rq);

		// --------------------------------------------------------------------
		// 1) Initialize the accumulation buffer sum[p] for this (r, q).
		//
		//    sum is 1D, indexed only by 'p'. We reuse the same physical
		//    buffer for every (r, q), but logically it is private to the
		//    current slice because we fully overwrite it here before use.
		// --------------------------------------------------------------------
		traverser(sum).for_each([&](auto state_p) {
			sum[state_p] = static_cast<num_t>(0);
		});

		// --------------------------------------------------------------------
		// 2) Accumulate:
		//        sum[p] += A[r][q][s] * C4[s][p]
		//
		//    We:
		//      * fix r and q in A to get a 1D view A_rq over its last dim;
		//      * rename that last dimension from 'p' to 's' so that we can
		//        treat it as the reduction dimension explicitly;
		//      * traverse the combined (s, p) space of (A_rq, C4, sum) with
		//        s outer and p inner, hoisting A[r][q][s] into a scalar and
		//        reusing it across the inner p-loop.
		// --------------------------------------------------------------------

		// View of A restricted to the current (r, q) plane; last dim is A[r,q,p].
		// We rename that 'p' dimension to 's', because in the math it is the
		// reduction index.
		auto A_rq = A
			^ fix<'r'>(r_idx)
			^ fix<'q'>(q_idx)
			^ rename<'p', 's'>();

		// Joint traverser over A_rq (dim 's'), C4 (dims 's','p') and sum (dim 'p').
		// The union of dimensions is {'s','p'}.
		traverser(A_rq, C4, sum).template for_dims<'s'>([&](auto trav_s) {
			// trav_s has 's' fixed, and still sees 'p' as the remaining dimension.
			// Load A[r][q][s] once and reuse it for all p.
			auto a_rqs = A_rq[trav_s];

			// Inner loop over contiguous p for this fixed s.
			trav_s.for_each([&](auto state_sp) {
				// state_sp carries both 's' and 'p'. Access patterns:
				//   - sum[state_sp]  -> sum[p]       (ignores 's')
				//   - C4[state_sp]   -> C4[s][p]
				sum[state_sp] += a_rqs * C4[state_sp];
			});
		});

		// --------------------------------------------------------------------
		// 3) Write back A[r][q][p] = sum[p] for all p.
		//
		//    We reuse trav_rq to iterate p for the current (r, q) slice of A.
		// --------------------------------------------------------------------
		trav_rq.template for_dims<'p'>([&](auto trav_rqp) {
			auto state_rqp = trav_rqp.state(); // contains 'r','q','p'
			A[state_rqp] = sum[state_rqp];     // sum only uses 'p'
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t nr = NR;
	std::size_t nq = NQ;
	std::size_t np = NP;

	// allocate data (bags)
	auto A = noarr::make_bag(
		noarr::scalar<num_t>()
		^ A_layout
		^ noarr::set_length<'r'>(nr)
		^ noarr::set_length<'q'>(nq)
		^ noarr::set_length<'p'>(np));

	auto C4 = noarr::make_bag(
		noarr::scalar<num_t>()
		^ C4_layout
		^ noarr::set_length<'s'>(np)
		^ noarr::set_length<'p'>(np));

	auto sum = noarr::make_bag(
		noarr::scalar<num_t>()
		^ sum_layout
		^ noarr::set_length<'p'>(np));

	// initialize arrays
	init_array(A.get_ref(), C4.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_doitgen(A.get_ref(), C4.get_ref(), sum.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// optionally print output (to prevent DCE and to check correctness)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'r' for a more natural outer dimension in the dump
		noarr::serialize_data(std::cerr, A.get_ref() ^ noarr::hoist<'r'>());
	}

	// print elapsed time in seconds
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}