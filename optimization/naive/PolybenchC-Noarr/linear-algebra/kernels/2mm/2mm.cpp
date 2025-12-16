#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions (problem sizes, DATA_TYPE, etc.)
#include "defines.hpp"

// include benchmark-specific definitions
#include "2mm.hpp"

using num_t = DATA_TYPE;

namespace {

// convenient dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();
constexpr auto l_vec = noarr::vector<'l'>();

// layout tuning: choose a concrete physical layout for each array
struct tuning {
	// tmp: i x j  (row-major: j inner, i outer)
	DEFINE_PROTO_STRUCT(tmp_layout, j_vec ^ i_vec);
	// A:   i x k  (row-major: k inner, i outer)
	DEFINE_PROTO_STRUCT(a_layout,  k_vec ^ i_vec);
	// B:   k x j  (row-major: j inner, k outer)
	DEFINE_PROTO_STRUCT(b_layout,  j_vec ^ k_vec);
	// C:   j x l  (row-major: l inner, j outer)
	DEFINE_PROTO_STRUCT(c_layout,  l_vec ^ j_vec);
	// D:   i x l  (row-major: l inner, i outer)
	DEFINE_PROTO_STRUCT(d_layout,  l_vec ^ i_vec);
} tuning;

// Array initialization.
//   A: ni x nk, index A[i][k]
//   B: nk x nj, index B[k][j]
//   C: nj x nl, index C[j][l]
//   D: ni x nl, index D[i][l]
void init_array(num_t &alpha, num_t &beta, auto A, auto B, auto C, auto D) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	const std::size_t ni = A | get_length<'i'>();
	const std::size_t nk = A | get_length<'k'>();
	const std::size_t nj = B | get_length<'j'>();
	const std::size_t nl = C | get_length<'l'>();

	// A[i][k] = ((i * k + 1) % ni) / ni
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)((i * k + 1) % ni) / (num_t)ni;
	});

	// B[k][j] = (k * (j + 1) % nj) / nj
	traverser(B).for_each([=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 1) % nj) / (num_t)nj;
	});

	// C[j][l] = ((j * (l + 3) + 1) % nl) / nl
	traverser(C).for_each([=](auto state) {
		auto [j, l] = get_indices<'j', 'l'>(state);
		C[state] = (num_t)((j * (l + 3) + 1) % nl) / (num_t)nl;
	});

	// D[i][l] = (i * (l + 2) % nk) / nk
	// note: uses nk as in the original C code
	traverser(D).for_each([=](auto state) {
		auto [i, l] = get_indices<'i', 'l'>(state);
		D[state] = (num_t)(i * (l + 2) % nk) / (num_t)nk;
	});
}

// Main computational kernel:
//   tmp  = alpha * A * B
//   D    = alpha * A * B * C + beta * D
// with shapes:
//   A:   i x k
//   B:   k x j
//   tmp: i x j
//   C:   j x l
//   D:   i x l
//
// Optimized traversal:
//  - Phase 1 (tmp = alpha * A * B)
//      for i
//        zero tmp[i,*]
//        for k
//          aik = alpha * A[i,k]
//          for j  (innermost, unit-stride in both tmp and B)
//            tmp[i,j] += aik * B[k,j]
//    This keeps j as the innermost loop, which is the contiguous
//    dimension for both tmp (i x j, row-major) and B (k x j, row-major),
//    and hoists A[i,k] and alpha out of the j-loop.
//
//  - Phase 2 (D = tmp * C + beta * D; note tmp already includes alpha)
//      for i
//        scale D[i,*] by beta (unit-stride in l)
//        for j
//          t = tmp[i,j]
//          for l  (innermost, unit-stride in both C and D)
//            D[i,l] += t * C[j,l]
//    Here l is the innermost loop, matching the contiguous dimension
//    of both C (j x l, row-major) and D (i x l, row-major).
[[gnu::flatten, gnu::noinline]]
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
	using namespace noarr;

#pragma scop
	// -----------------------------------------------------------------
	// Phase 1: tmp[i][j] = alpha * sum_k A[i][k] * B[k][j]
	// -----------------------------------------------------------------
	traverser(tmp, A, B).template for_dims<'i'>([=](auto trav_i) {
		// trav_i: 'i' fixed, 'j' and 'k' still free.

		// 1) Zero whole row tmp[i, *] (contiguous in 'j').
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			// trav_ij: 'i' and 'j' fixed, 'k' free (unused here).
			tmp[trav_ij] = (num_t)0;
		});

		// 2) For each k, update the whole row tmp[i, *].
		trav_i.template for_dims<'k'>([=](auto trav_ik) {
			// trav_ik: 'i' and 'k' fixed, 'j' free.

			// Hoist A[i,k] * alpha out of the j-loop.
			const num_t aik_scaled = alpha * A[trav_ik];

			// With i and k fixed, only 'j' remains free.
			// 'j' is the innermost/contiguous dimension of both tmp and B,
			// so this loop has unit-stride accesses in both matrices.
			trav_ik.template for_dims<'j'>([=](auto trav_ikj) {
				tmp[trav_ikj] += aik_scaled * B[trav_ikj];
			});
		});
	});

	// -----------------------------------------------------------------
	// Phase 2: D[i][l] = beta * D[i][l] + sum_j tmp[i][j] * C[j][l]
	//            (tmp already includes the factor alpha)
	// -----------------------------------------------------------------
	traverser(D, tmp, C).template for_dims<'i'>([=](auto trav_i) {
		// trav_i: 'i' fixed, 'j' and 'l' still free.

		// 1) Scale the entire row D[i, *] by beta (unit-stride in 'l').
		trav_i.template for_dims<'l'>([=](auto trav_il) {
			// trav_il: 'i' and 'l' fixed, 'j' free (unused here).
			D[trav_il] *= beta;
		});

		// 2) Accumulate tmp[i, j] * C[j, l] into D[i, l].
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			// trav_ij: 'i' and 'j' fixed, 'l' free.

			// Load tmp[i, j] once and reuse it across the l-loop.
			const num_t tmp_ij = tmp[trav_ij];

			// With i and j fixed, only 'l' remains free.
			// 'l' is the innermost/contiguous dimension of both C and D,
			// so this loop is unit-stride in both arrays and vectorizes well.
			trav_ij.template for_dims<'l'>([=](auto trav_ijl) {
				D[trav_ijl] += tmp_ij * C[trav_ijl];
			});
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem sizes
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;

	// scalars
	num_t alpha;
	num_t beta;

	// set all potential lengths once; each layout will use the relevant ones
	auto set_lengths =
		noarr::set_length<'i'>(ni) ^
		noarr::set_length<'j'>(nj) ^
		noarr::set_length<'k'>(nk) ^
		noarr::set_length<'l'>(nl);

	// allocate bags with given layouts and lengths
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout   ^ set_lengths);
	auto B   = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout   ^ set_lengths);
	auto C   = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout   ^ set_lengths);
	auto D   = noarr::bag(noarr::scalar<num_t>() ^ tuning.d_layout   ^ set_lengths);

	// initialize arrays
	init_array(alpha, beta, A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	// measure execution time
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_2mm(alpha, beta, tmp.get_ref(), A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// optionally dump the resulting D matrix (layout-agnostic)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'i' to be the outer dimension for row-wise printing
		noarr::serialize_data(std::cerr, D.get_ref() ^ noarr::hoist<'i'>());
	}

	// print elapsed time (seconds)
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}