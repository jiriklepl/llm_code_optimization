#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "3mm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();
constexpr auto l_vec = noarr::vector<'l'>();
constexpr auto m_vec = noarr::vector<'m'>();

struct tuning {
	// A: i x k
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: k x j
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);
	// C: j x m
	DEFINE_PROTO_STRUCT(c_layout, m_vec ^ j_vec);
	// D: m x l
	DEFINE_PROTO_STRUCT(d_layout, l_vec ^ m_vec);
	// E: i x j
	DEFINE_PROTO_STRUCT(e_layout, j_vec ^ i_vec);
	// F: j x l
	DEFINE_PROTO_STRUCT(f_layout, l_vec ^ j_vec);
	// G: i x l
	DEFINE_PROTO_STRUCT(g_layout, l_vec ^ i_vec);
} tuning;

// Array initialization
void init_array(auto A, auto B, auto C, auto D) {
	// A: i x k
	// B: k x j
	// C: j x m
	// D: m x l
	using namespace noarr;

	auto ni = A | get_length<'i'>();
	auto nk = A | get_length<'k'>();
	auto nj = B | get_length<'j'>();
	auto nm = C | get_length<'m'>();
	auto nl = D | get_length<'l'>();

	// A[i][k] = ((i * k + 1) % ni) / (5 * ni)
	traverser(A).template for_dims<'i', 'k'>([=](auto trav) {
		auto state = trav.state();
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)((i * k + 1) % ni) / (5 * ni);
	});

	// B[k][j] = ((k * (j + 1) + 2) % nj) / (5 * nj)
	traverser(B).template for_dims<'k', 'j'>([=](auto trav) {
		auto state = trav.state();
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)((k * (j + 1) + 2) % nj) / (5 * nj);
	});

	// C[j][m] = (j * (m + 3) % nl) / (5 * nl)
	traverser(C).template for_dims<'j', 'm'>([=](auto trav) {
		auto state = trav.state();
		auto [j, m] = get_indices<'j', 'm'>(state);
		C[state] = (num_t)((j * (m + 3) % nl)) / (5 * nl);
	});

	// D[m][l] = ((m * (l + 2) + 2) % nk) / (5 * nk)
	traverser(D).template for_dims<'m', 'l'>([=](auto trav) {
		auto state = trav.state();
		auto [m, l] = get_indices<'m', 'l'>(state);
		D[state] = (num_t)(((m * (l + 2) + 2) % nk)) / (5 * nk);
	});
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
	// E: i x j  = A (i x k) * B (k x j)
	// F: j x l  = C (j x m) * D (m x l)
	// G: i x l  = E (i x j) * F (j x l)
	using namespace noarr;

	#pragma scop
	// E := A * B
	traverser(E, A, B).template for_dims<'i', 'j'>([=](auto trav_ij) {
		// initialize E[i][j] = 0
		E[trav_ij] = (num_t)0.0;

		// accumulate over k
		trav_ij.template for_dims<'k'>([=](auto trav_ijk) {
			E[trav_ijk] += A[trav_ijk] * B[trav_ijk];
		});
	});

	// F := C * D
	traverser(F, C, D).template for_dims<'j', 'l'>([=](auto trav_jl) {
		// initialize F[j][l] = 0
		F[trav_jl] = (num_t)0.0;

		// accumulate over m
		trav_jl.template for_dims<'m'>([=](auto trav_jlm) {
			F[trav_jlm] += C[trav_jlm] * D[trav_jlm];
		});
	});

	// G := E * F
	traverser(G, E, F).template for_dims<'i', 'l'>([=](auto trav_il) {
		// initialize G[i][l] = 0
		G[trav_il] = (num_t)0.0;

		// accumulate over j
		trav_il.template for_dims<'j'>([=](auto trav_ilj) {
			G[trav_ilj] += E[trav_ilj] * F[trav_ilj];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;
	std::size_t nm = NM;

	// set lengths for all dimensions
	auto set_lengths =
		noarr::set_length<'i'>(ni) ^
		noarr::set_length<'j'>(nj) ^
		noarr::set_length<'k'>(nk) ^
		noarr::set_length<'l'>(nl) ^
		noarr::set_length<'m'>(nm);

	// allocate data
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto D = noarr::bag(noarr::scalar<num_t>() ^ tuning.d_layout ^ set_lengths);
	auto E = noarr::bag(noarr::scalar<num_t>() ^ tuning.e_layout ^ set_lengths);
	auto F = noarr::bag(noarr::scalar<num_t>() ^ tuning.f_layout ^ set_lengths);
	auto G = noarr::bag(noarr::scalar<num_t>() ^ tuning.g_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_3mm(E.get_ref(), A.get_ref(), B.get_ref(),
	           F.get_ref(), C.get_ref(), D.get_ref(), G.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (to prevent dead-code elimination)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, G.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}