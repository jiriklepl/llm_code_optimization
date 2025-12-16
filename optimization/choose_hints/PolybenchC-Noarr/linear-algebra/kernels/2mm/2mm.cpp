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
// The implementation below is organized to improve cache locality:
//
// 1) For tmp = alpha * A * B we iterate i, then k, then j:
//      for i
//        for k
//          aik = alpha * A(i,k)
//          for j
//            tmp(i,j) += aik * B(k,j)
//    This makes accesses to A (i,k), B (k,j) and tmp(i,j) all contiguous
//    in their inner dimension.
//
// 2) For D = beta * D + tmp * C we first scale D by beta, then iterate
//    i, then j, then l:
//      for i
//        for j
//          tmp_ij = tmp(i,j)
//          for l
//            D(i,l) += tmp_ij * C(j,l)
//    This makes accesses to C(j,l) and D(i,l) contiguous in l while
//    reading tmp(i,j) once per (i,j).
[[gnu::flatten, gnu::noinline]]
void kernel_2mm(num_t alpha, num_t beta, auto tmp, auto A, auto B, auto C, auto D) {
	using namespace noarr;

#pragma scop
	// ------------------------------------------------------------------
	// Phase 0: initialize tmp[i][j] = 0
	// ------------------------------------------------------------------
	traverser(tmp).for_each([=](auto s) {
		tmp[s] = (num_t)0;
	});

	// ------------------------------------------------------------------
	// Phase 1: tmp[i][j] += alpha * sum_k A[i][k] * B[k][j]
	//
	// Loop order: i (outer) -> k (middle) -> j (inner)
	//
	// Layout considerations:
	//   - A has dimensions (i x k) with k inner  -> k varying in innermost
	//     loop yields contiguous A[i][k].
	//   - B has dimensions (k x j) with j inner  -> for fixed k, varying j
	//     yields contiguous B[k][j].
	//   - tmp has dimensions (i x j) with j inner -> for fixed i, varying j
	//     yields contiguous tmp[i][j].
	// ------------------------------------------------------------------
	traverser(tmp, A, B).template for_dims<'i'>([=](auto trav_i) {
		// i is fixed here

		// Iterate over k next to keep A[i][k] contiguous
		trav_i.template for_dims<'k'>([=](auto trav_ik) {
			// i and k are fixed here

			// Alpha-scaled A(i,k) is invariant across j, so hoist it
			const num_t aik_scaled = alpha * A[trav_ik];

			// Inner-most loop over j: tmp(i, j) and B(k, j) contiguous
			trav_ik.template for_dims<'j'>([=](auto trav_ikj) {
				tmp[trav_ikj] += aik_scaled * B[trav_ikj];
			});
		});
	});

	// ------------------------------------------------------------------
	// Phase 2a: scale D by beta
	//
	// D[i][l] = beta * D[i][l]
	// ------------------------------------------------------------------
	traverser(D).for_each([=](auto s) {
		D[s] *= beta;
	});

	// ------------------------------------------------------------------
	// Phase 2b: accumulate D[i][l] += sum_j tmp[i][j] * C[j][l]
	//
	// Loop order: i (outer) -> j (middle) -> l (inner)
	//
	// Layout considerations:
	//   - tmp has dimensions (i x j) with j inner  -> for fixed i,
	//     iterating j is cache-friendly when streaming tmp row-wise.
	//   - C has dimensions (j x l) with l inner    -> for fixed j,
	//     iterating l yields contiguous C[j][l].
	//   - D has dimensions (i x l) with l inner    -> for fixed i,
	//     iterating l yields contiguous D[i][l].
	//
	// We read tmp(i,j) once per (i,j), then update an entire row of D
	// using the corresponding row of C.
	// ------------------------------------------------------------------
	traverser(D, tmp, C).template for_dims<'i'>([=](auto trav_i) {
		// i is fixed here

		// Iterate over j next: read tmp(i,j) in its contiguous direction
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			// i and j are fixed here

			const num_t tmp_ij = tmp[trav_ij];

			// Inner-most loop over l: C(j,l) and D(i,l) contiguous
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