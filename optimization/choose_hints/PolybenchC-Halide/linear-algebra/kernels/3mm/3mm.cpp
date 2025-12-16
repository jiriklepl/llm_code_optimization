#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "3mm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();
constexpr auto l_vec = noarr::vector<'l'>();
constexpr auto m_vec = noarr::vector<'m'>();

struct tuning {
	// E: i x j
	static constexpr auto e_layout = j_vec ^ i_vec;
	// A: i x k
	static constexpr auto a_layout = k_vec ^ i_vec;
	// B: k x j
	static constexpr auto b_layout = j_vec ^ k_vec;
	// F: j x l
	static constexpr auto f_layout = l_vec ^ j_vec;
	// C: j x m
	static constexpr auto c_layout = m_vec ^ j_vec;
	// D: m x l
	static constexpr auto d_layout = l_vec ^ m_vec;
	// G: i x l
	static constexpr auto g_layout = l_vec ^ i_vec;
} tuning;

// initialization function
void init_array(int ni, int nj, int nk, int nl, int nm, auto A, auto B, auto C, auto D) {
	using namespace noarr;

	traverser(A) | [=](auto s) {
		auto [i, k] = get_indices<'i', 'k'>(s);
		A[s] = (num_t)(((i * k + 1) % ni)) / (num_t)(5 * ni);
	};

	traverser(B) | [=](auto s) {
		auto [k, j] = get_indices<'k', 'j'>(s);
		B[s] = (num_t)(((k * (j + 1) + 2) % nj)) / (num_t)(5 * nj);
	};

	traverser(C) | [=](auto s) {
		auto [j, m] = get_indices<'j', 'm'>(s);
		C[s] = (num_t)((j * (m + 3) % nl)) / (num_t)(5 * nl);
	};

	traverser(D) | [=](auto s) {
		auto [m, l] = get_indices<'m', 'l'>(s);
		D[s] = (num_t)(((m * (l + 2) + 2) % nk)) / (num_t)(5 * nk);
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
	using namespace noarr;

	#pragma scop
	// E := A * B
	traverser(E, A, B) | for_dims<'i'>([=](auto t_i) {
		t_i | for_each<'j'>([=](auto s) {
			E[s] = (num_t)0;
		});
		t_i | for_dims<'k'>([=](auto t_k) {
			t_k | for_each<'j'>([=](auto s) {
				E[s] += A[s] * B[s];
			});
		});
	});

	// F := C * D
	traverser(F, C, D) | for_dims<'j'>([=](auto t_j) {
		t_j | for_each<'l'>([=](auto s) {
			F[s] = (num_t)0;
		});
		t_j | for_dims<'m'>([=](auto t_m) {
			t_m | for_each<'l'>([=](auto s) {
				F[s] += C[s] * D[s];
			});
		});
	});

	// G := E * F
	traverser(G, E, F) | for_dims<'i'>([=](auto t_i) {
		t_i | for_each<'l'>([=](auto s) {
			G[s] = (num_t)0;
		});
		t_i | for_dims<'j'>([=](auto t_j) {
			t_j | for_each<'l'>([=](auto s) {
				G[s] += E[s] * F[s];
			});
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

	auto set_lengths =
		noarr::set_length<'i'>(ni) ^
		noarr::set_length<'j'>(nj) ^
		noarr::set_length<'k'>(nk) ^
		noarr::set_length<'l'>(nl) ^
		noarr::set_length<'m'>(nm);

	// bags
	auto E = noarr::bag(noarr::scalar<num_t>() ^ tuning.e_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);
	auto F = noarr::bag(noarr::scalar<num_t>() ^ tuning.f_layout ^ set_lengths);
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto D = noarr::bag(noarr::scalar<num_t>() ^ tuning.d_layout ^ set_lengths);
	auto G = noarr::bag(noarr::scalar<num_t>() ^ tuning.g_layout ^ set_lengths);

	// initialize data
	init_array((int)ni, (int)nj, (int)nk, (int)nl, (int)nm, A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_3mm(E.get_ref(), A.get_ref(), B.get_ref(), F.get_ref(), C.get_ref(), D.get_ref(), G.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (DCE prevention)
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, G.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
