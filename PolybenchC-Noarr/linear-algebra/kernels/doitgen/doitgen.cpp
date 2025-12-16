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
[[gnu::flatten, gnu::noinline]]
void kernel_doitgen(auto A, auto C4, auto sum) {
	using namespace noarr;

	// C kernel:
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
	traverser(A, sum).template for_dims<'r', 'q'>([=](auto trav_rq) {

		// First loop over p: compute sum[p] = Σ_s A[r][q][s] * C4[s][p]
		trav_rq.template for_dims<'p'>([=](auto trav_rqp) {
			auto state_rqp = trav_rqp.state(); // contains indices r, q, p

			// sum[p] = 0.0;
			sum[state_rqp] = static_cast<num_t>(0);

			// extract indices r, q, p
			auto r_idx = get_index<'r'>(state_rqp);
			auto q_idx = get_index<'q'>(state_rqp);
			auto p_idx = get_index<'p'>(state_rqp);

			// view of C4 with fixed column p_idx: C4[s][p_idx]
			auto C4_p = C4 ^ noarr::fix<'p'>(p_idx);

			// inner loop over s
			noarr::traverser(C4_p).for_each([=](auto state_s) {
				auto s_idx = get_index<'s'>(state_s);

				// A[r][q][s]
				auto state_rqs = noarr::idx<'r', 'q', 'p'>(r_idx, q_idx, s_idx);

				// sum[p] += A[r][q][s] * C4[s][p]
				sum[state_rqp] += A[state_rqs] * C4_p[state_s];
			});
		});

		// Second loop over p: A[r][q][p] = sum[p]
		trav_rq.template for_dims<'p'>([=](auto trav_rqp) {
			auto state_rqp = trav_rqp.state();
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