#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "doitgen.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto r_vec = noarr::vector<'r'>();
constexpr auto q_vec = noarr::vector<'q'>();
constexpr auto p_vec = noarr::vector<'p'>();
constexpr auto s_vec = noarr::vector<'s'>();

struct tuning {
	// A: r x q x p, contiguous in p
	DEFINE_PROTO_STRUCT(a_layout, p_vec ^ q_vec ^ r_vec);
	// C4: s x p, contiguous in p (row s, column p)
	DEFINE_PROTO_STRUCT(c4_layout, p_vec ^ s_vec);
	// sum: p
	DEFINE_PROTO_STRUCT(sum_layout, p_vec);
} tuning;

// initialization function
void init_array(auto A, auto C4) {
	using namespace noarr;

	traverser(A) | [=](auto state) {
		auto [r, q, p] = get_indices<'r', 'q', 'p'>(state);
		std::size_t np = A | get_length<'p'>();
		A[state] = (num_t)(((r * q + p) % np)) / (num_t)np;
	};

	traverser(C4) | [=](auto state) {
		auto [s, p] = get_indices<'s', 'p'>(state);
		std::size_t np = C4 | get_length<'p'>();
		C4[state] = (num_t)(((s * p) % np)) / (num_t)np;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_doitgen(auto A, auto C4, auto sum) {
	using namespace noarr;

	// for (r) for (q) { for (p) sum[p]=0; for (p) for (s) sum[p]+=A[r][q][s]*C4[s][p]; for (p) A[r][q][p]=sum[p]; }
	#pragma scop
	traverser(A, C4, sum) | for_dims<'r'>([=](auto trav_r) {

		trav_r | for_dims<'q'>([=](auto trav_rq) {

			// sum[p] = 0
			trav_rq | for_each<'p'>([=](auto st_rqp) {
				sum[st_rqp] = (num_t)0;
			});

			// for each p, accumulate over s: sum[p] += A[r][q][s] * C4[s][p]
			trav_rq | for_dims<'p'>([=](auto trav_rqp) {
				trav_rqp | for_each<'s'>([=](auto st_rqps) {
					// A expects its 'p' index to be 's' (A[r][q][s]), override 'p' in the state with current 's'
					auto st_for_A = st_rqps.template with<noarr::index_in<'p'>>(noarr::get_index<'s'>(st_rqps));
					sum[st_rqps] += A[st_for_A] * C4[st_rqps];
				});
			});

			// A[r][q][p] = sum[p]
			trav_rq | for_each<'p'>([=](auto st_rqp) {
				A[st_rqp] = sum[st_rqp];
			});
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

	// set common lengths
	auto set_lengths = noarr::set_length<'r'>(nr) ^ noarr::set_length<'q'>(nq) ^ noarr::set_length<'p'>(np) ^ noarr::set_length<'s'>(np);

	// bags
	auto A  = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout  ^ set_lengths);
	auto C4 = noarr::bag(noarr::scalar<num_t>() ^ tuning.c4_layout ^ set_lengths);
	auto sum = noarr::bag(noarr::scalar<num_t>() ^ tuning.sum_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), C4.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_doitgen(A.get_ref(), C4.get_ref(), sum.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		// hoist r and q for a predictable dump order
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'r'>() ^ noarr::hoist<'q'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
