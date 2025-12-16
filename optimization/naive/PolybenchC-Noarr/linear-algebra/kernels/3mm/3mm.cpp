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

	// Precompute constants used in initialization to avoid repeated divisions
	const num_t inv_5ni = num_t(1.0) / num_t(5.0 * static_cast<num_t>(ni));
	const num_t inv_5nj = num_t(1.0) / num_t(5.0 * static_cast<num_t>(nj));
	const num_t inv_5nl = num_t(1.0) / num_t(5.0 * static_cast<num_t>(nl));
	const num_t inv_5nk = num_t(1.0) / num_t(5.0 * static_cast<num_t>(nk));

	// A[i][k] = ((i * k + 1) % ni) / (5 * ni)
	traverser(A).template for_dims<'i', 'k'>([&](auto trav) {
		auto state = trav.state();
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = num_t((i * k + 1) % ni) * inv_5ni;
	});

	// B[k][j] = ((k * (j + 1) + 2) % nj) / (5 * nj)
	traverser(B).template for_dims<'k', 'j'>([&](auto trav) {
		auto state = trav.state();
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = num_t((k * (j + 1) + 2) % nj) * inv_5nj;
	});

	// C[j][m] = (j * (m + 3) % nl) / (5 * nl)
	traverser(C).template for_dims<'j', 'm'>([&](auto trav) {
		auto state = trav.state();
		auto [j, m] = get_indices<'j', 'm'>(state);
		C[state] = num_t((j * (m + 3)) % nl) * inv_5nl;
	});

	// D[m][l] = ((m * (l + 2) + 2) % nk) / (5 * nk)
	traverser(D).template for_dims<'m', 'l'>([&](auto trav) {
		auto state = trav.state();
		auto [m, l] = get_indices<'m', 'l'>(state);
		D[state] = num_t((m * (l + 2) + 2) % nk) * inv_5nk;
	});
}

// computation kernel
// NOTE: All loop nests are expressed using Noarr traversers and for_dims.
//       The multiplications are reordered to improve cache locality while
//       preserving the mathematical semantics of the original 3mm kernel.
[[gnu::flatten, gnu::noinline]]
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
	// E: i x j  = A (i x k) * B (k x j)
	// F: j x l  = C (j x m) * D (m x l)
	// G: i x l  = E (i x j) * F (j x l)
	using namespace noarr;

	#pragma scop

	// ---------------------------------------------------------------------
	// E := A * B
	// Original order: for i, j, k
	// New order:      for i, k, j
	//
	// Data layout:
	//   A: dims 'i', 'k'  (k contiguous)
	//   B: dims 'k', 'j'  (j contiguous)
	//   E: dims 'i', 'j'  (j contiguous)
	//
	// New loop order makes inner loop contiguous in j for B and E,
	// while reading each A(i,k) once and reusing it across the j-loop.
	// ---------------------------------------------------------------------

	// Initialize E to zero
	traverser(E).for_each([&](auto s) {
		E[s] = num_t(0.0);
	});

	// Accumulate E = A * B
	traverser(E, A, B).template for_dims<'i', 'k'>([&](auto trav_ik) {
		// trav_ik has 'i' and 'k' fixed; no 'j' yet.

		// Load A(i,k) once and reuse inside the j-loop
		const num_t aik = A[trav_ik];

		// Now iterate over j for this (i,k) pair
		trav_ik.template for_dims<'j'>([&](auto trav_ikj) {
			// state has (i,k,j); B uses (k,j), E uses (i,j).
			E[trav_ikj] += aik * B[trav_ikj];
		});
	});

	// ---------------------------------------------------------------------
	// F := C * D
	// Original order: for j, l, m
	// New order:      for j, m, l
	//
	// Data layout:
	//   C: dims 'j', 'm'  (m contiguous)
	//   D: dims 'm', 'l'  (l contiguous)
	//   F: dims 'j', 'l'  (l contiguous)
	//
	// New loop order:
	//   - walks C row-by-row in m (good locality),
	//   - for each (j,m) multiplies whole row D(m, :) into F(j, :),
	//   - inner l-loop is contiguous for both D and F.
	// ---------------------------------------------------------------------

	// Initialize F to zero
	traverser(F).for_each([&](auto s) {
		F[s] = num_t(0.0);
	});

	// Accumulate F = C * D
	traverser(F, C, D).template for_dims<'j', 'm'>([&](auto trav_jm) {
		// trav_jm has 'j' and 'm' fixed.

		// Load C(j,m) once and reuse across the l-loop
		const num_t cjm = C[trav_jm];

		// Iterate over l for this (j,m) pair
		trav_jm.template for_dims<'l'>([&](auto trav_jml) {
			// state has (j,m,l); D uses (m,l), F uses (j,l).
			F[trav_jml] += cjm * D[trav_jml];
		});
	});

	// ---------------------------------------------------------------------
	// G := E * F
	// Original order: for i, l, j
	// New order:      for i, j, l
	//
	// Data layout:
	//   E: dims 'i', 'j'  (j contiguous)
	//   F: dims 'j', 'l'  (l contiguous)
	//   G: dims 'i', 'l'  (l contiguous)
	//
	// New loop order:
	//   - walks E row-by-row in j (good locality),
	//   - for each (i,j) accumulates E(i,j) * F(j, :) into G(i, :),
	//   - inner l-loop is contiguous for both F and G.
	// ---------------------------------------------------------------------

	// Initialize G to zero
	traverser(G).for_each([&](auto s) {
		G[s] = num_t(0.0);
	});

	// Accumulate G = E * F
	traverser(G, E, F).template for_dims<'i', 'j'>([&](auto trav_ij) {
		// trav_ij has 'i' and 'j' fixed.

		// Load E(i,j) once and reuse across the l-loop
		const num_t eij = E[trav_ij];

		// Iterate over l for this (i,j) pair
		trav_ij.template for_dims<'l'>([&](auto trav_ijl) {
			// state has (i,j,l); F uses (j,l), G uses (i,l).
			G[trav_ijl] += eij * F[trav_ijl];
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