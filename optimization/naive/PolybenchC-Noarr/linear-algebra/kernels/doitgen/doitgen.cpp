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
constexpr auto A_layout   = p_vec ^ q_vec ^ r_vec; // A[r][q][p]
constexpr auto C4_layout  = p_vec ^ s_vec;         // C4[s][p]
constexpr auto sum_layout = p_vec;                 // sum[p]

// initialization function
void init_array(auto A, auto C4) {
	using namespace noarr;

	// A[r][q][p] = ((r*q + p) % np) / np;
	traverser(A).for_each([=](auto state) {
		auto [r, q, p] = get_indices<'r', 'q', 'p'>(state);
		auto np_len = A | get_length<'p'>();

		A[state] = static_cast<num_t>(((r * q + p) % np_len))
			/ static_cast<num_t>(np_len);
	});

	// C4[s][p] = (s*p % np) / np;
	traverser(C4).for_each([=](auto state) {
		auto [s, p] = get_indices<'s', 'p'>(state);
		auto np_len = C4 | get_length<'p'>();

		C4[state] = static_cast<num_t>(((s * p) % np_len))
			/ static_cast<num_t>(np_len);
	});
}

// computation kernel
//
// Original scalar kernel (C-like pseudocode), as given in the comment:
//
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
//
// We implement an equivalent, but more locality-friendly ordering:
//
//   for (r)
//     for (q) {
//       for (p) sum[p] = 0;
//       for (s) {
//         const a = A[r][q][s];
//         for (p)
//           sum[p] += a * C4[s][p];
//       }
//       for (p) A[r][q][p] = sum[p];
//     }
//
// For each fixed (r,q,p), the sequence of additions over s is still s = 0..NP-1,
// so floating-point accumulation order per output element is preserved.
// The main improvements:
//
//   * A[r][q][s] is loaded once per s (per (r,q)), instead of once per (p,s).
//   * C4[s][p] and sum[p] are traversed with p as the innermost dimension,
//     matching their memory layout (p is the fastest-varying index).
//   * All loop nests are expressed via Noarr traversers.
[[gnu::flatten, gnu::noinline]]
void kernel_doitgen(auto A, auto C4, auto sum) {
	using namespace noarr;

#pragma scop
	// Outer loops over r and q – traverse A and sum together so that
	// we can reuse the same (r,q,p) states when writing back.
	traverser(A, sum).template for_dims<'r', 'q'>([&](auto trav_rq) {
		// trav_rq has 'r' and 'q' fixed; only 'p' remains free inside trav_rq.for_each().
		auto state_rq = trav_rq.state();
		auto r_idx = get_index<'r'>(state_rq);
		auto q_idx = get_index<'q'>(state_rq);

		// 1) Initialize sum[p] = 0 for all p.
		//    sum depends only on 'p', so this touch is perfectly contiguous.
		traverser(sum).for_each([&](auto state_p) {
			sum[state_p] = static_cast<num_t>(0);
		});

		// 2) Accumulate sum[p] = Σ_s A[r][q][s] * C4[s][p].
		//
		// We iterate C4 with 's' as an explicit loop dimension and
		// 'p' as the innermost loop, so accesses to C4[s][p] and sum[p]
		// follow the contiguous 'p' dimension.
		traverser(C4).template for_dims<'s'>([&](auto trav_s) {
			// trav_s has 's' fixed; 'p' is left free.
			auto state_s = trav_s.state();
			auto s_idx = get_index<'s'>(state_s);

			// Load A[r][q][s] once for this s and (r,q), then reuse it for all p.
			auto state_rqs = idx<'r', 'q', 'p'>(r_idx, q_idx, s_idx);
			const num_t a_rqs = A[state_rqs];

			// Inner loop over p: contiguous over the 'p' dimension.
			trav_s.for_each([&](auto state_sp) {
				// state_sp carries 's' and 'p'. The 's' index is ignored by sum.
				sum[state_sp] += a_rqs * C4[state_sp];
			});
		});

		// 3) Write back A[r][q][p] = sum[p] for all p.
		//
		// Here we reuse trav_rq, which has 'r' and 'q' fixed and iterates 'p';
		// the state carries ('r','q','p'), but sum only uses 'p' (the extra
		// indices are ignored by Noarr).
		trav_rq.for_each([&](auto state_rqp) {
			A[state_rqp] = sum[state_rqp];
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