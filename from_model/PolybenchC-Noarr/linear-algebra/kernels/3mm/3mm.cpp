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
	// Layout notes:
	// - We keep A, C, E, G row-major in their column index (good for left operand).
	// - We store B, D, F "column-major" w.r.t. the reduction dimension (k, m, j),
	//   so that in each matrix product both operands are contiguous along the
	//   inner reduction dimension. This improves cache locality and SIMD-friendliness.

	// A: i x k, row-major over k  -> contiguous k for fixed i
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);

	// B: k x j, column-major over k -> contiguous k for fixed j
	// (layout: dimensions 'j' outer, 'k' inner)
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ j_vec);

	// C: j x m, row-major over m -> contiguous m for fixed j
	DEFINE_PROTO_STRUCT(c_layout, m_vec ^ j_vec);

	// D: m x l, column-major over m -> contiguous m for fixed l
	// (layout: dimensions 'l' outer, 'm' inner)
	DEFINE_PROTO_STRUCT(d_layout, m_vec ^ l_vec);

	// E: i x j, row-major over j -> contiguous j for fixed i
	DEFINE_PROTO_STRUCT(e_layout, j_vec ^ i_vec);

	// F: j x l, column-major over j -> contiguous j for fixed l
	// (layout: dimensions 'l' outer, 'j' inner)
	DEFINE_PROTO_STRUCT(f_layout, j_vec ^ l_vec);

	// G: i x l, row-major over l -> contiguous l for fixed i
	DEFINE_PROTO_STRUCT(g_layout, l_vec ^ i_vec);
} tuning;

// Array initialization
//
// This function matches the original mathematical initialization but
// improves a couple of aspects:
//   * loop orders follow the physical layouts (unit-stride inner loops),
//   * per-matrix constants 1/(5 * N) are precomputed once instead of
//     recomputed in every iteration.
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

	// Precompute reciprocals used in initialization to avoid
	// repeated floating-point divisions.
	const num_t inv_5_ni = num_t(1) / num_t(5 * ni);
	const num_t inv_5_nj = num_t(1) / num_t(5 * nj);
	const num_t inv_5_nk = num_t(1) / num_t(5 * nk);
	const num_t inv_5_nl = num_t(1) / num_t(5 * nl);

	// A[i][k] = ((i * k + 1) % ni) / (5 * ni)
	// Traverse in (i, k) so that k (the inner, contiguous dim) is innermost.
	traverser(A).template for_dims<'i', 'k'>([=](auto trav) {
		auto state = trav.state();
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = num_t((i * k + 1) % ni) * inv_5_ni;
	});

	// B[k][j] = ((k * (j + 1) + 2) % nj) / (5 * nj)
	// B layout is (j outer, k inner), so we traverse (j, k) to keep k unit-stride.
	traverser(B).template for_dims<'j', 'k'>([=](auto trav) {
		auto state = trav.state();
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = num_t((k * (j + 1) + 2) % nj) * inv_5_nj;
	});

	// C[j][m] = (j * (m + 3) % nl) / (5 * nl)
	// Note: the original formula uses nl (length of 'l'), not nm.
	traverser(C).template for_dims<'j', 'm'>([=](auto trav) {
		auto state = trav.state();
		auto [j, m] = get_indices<'j', 'm'>(state);
		C[state] = num_t((j * (m + 3) % nl)) * inv_5_nl;
	});

	// D[m][l] = ((m * (l + 2) + 2) % nk) / (5 * nk)
	// D layout is (l outer, m inner), so we traverse (l, m) so that m is unit-stride.
	traverser(D).template for_dims<'l', 'm'>([=](auto trav) {
		auto state = trav.state();
		auto [m, l] = get_indices<'m', 'l'>(state);
		D[state] = num_t((m * (l + 2) + 2) % nk) * inv_5_nk;
	});
}

// computation kernel
//
// Original semantics:
//   E = A * B   (E: i x j,  A: i x k,  B: k x j)
//   F = C * D   (F: j x l,  C: j x m,  D: m x l)
//   G = E * F   (G: i x l,  E: i x j,  F: j x l)
//
// Here we keep the same mathematics but:
//
//   * choose layouts for B, D, F so that in each multiply both operands are
//     contiguous along the reduction dimension;
//   * apply 3D blocking with into_blocks_dynamic to improve cache reuse,
//     handling edge tiles correctly;
//   * use register-resident accumulators so each output element is written
//     only once, after its full dot-product is computed.
//
[[gnu::flatten, gnu::noinline]]
void kernel_3mm(auto E, auto A, auto B, auto F, auto C, auto D, auto G) {
	using namespace noarr;

	// Tile sizes: these are conservative, cache-friendly defaults
	// for a modern x64 CPU; they can be tuned if desired.
	constexpr std::size_t tile_i = 32;
	constexpr std::size_t tile_j = 32;
	constexpr std::size_t tile_k = 32;
	constexpr std::size_t tile_m = 32;
	constexpr std::size_t tile_l = 32;

	#pragma scop
	// -----------------------------------------------------------------
	// Stage 1: E = A * B
	//   E[i, j] = sum_k A[i, k] * B[k, j]
	//
	// We block in (i, j, k). For each (i, j) inside an (I, J) tile, we
	// compute a full dot-product over all k, accumulating into a scalar
	// and writing E[i, j] once at the end.
	// -----------------------------------------------------------------
	traverser(E, A, B)
		.order(
			// into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>
			// splits Dim into tiles (DimMajor, DimMinor) and introduces
			// a guard dimension DimIsPresent to safely handle edge tiles.
			into_blocks_dynamic<'i', 'I', 'i', 's'>(noarr::lit<tile_i>) ^
			into_blocks_dynamic<'j', 'J', 'j', 't'>(noarr::lit<tile_j>) ^
			into_blocks_dynamic<'k', 'K', 'k', 'u'>(noarr::lit<tile_k>) ^
			// Hoist tile indices so that (I, J, K) become outermost.
			hoist<'I'>() ^ hoist<'J'>() ^ hoist<'K'>()
		)
		// Iterate over tiles in (I, J). Each inner traverser trav_IJ has
		// a fixed tile of (i, j).
		.template for_dims<'I', 'J'>([=](auto trav_IJ) {
			// For each (i, j) in this tile, compute one full dot-product
			// over k. Indices 's' and 't' are presence guards for
			// possibly incomplete tiles and ensure we skip out-of-range
			// rows/columns automatically.
			trav_IJ.template for_dims<'i', 's', 'j', 't'>([=](auto trav_ij) {
				num_t acc = num_t(0);

				// Accumulate contributions from all (K, k) tiles.
				// 'u' is the guard that filters out invalid k in the
				// last partial block (if NK is not a multiple of tile_k).
				trav_ij.template for_dims<'K', 'k', 'u'>([&](auto trav_ijk) {
					acc += A[trav_ijk] * B[trav_ijk];
				});

				// Store the final dot-product once.
				E[trav_ij] = acc;
			});
		});

	// -----------------------------------------------------------------
	// Stage 2: F = C * D
	//   F[j, l] = sum_m C[j, m] * D[m, l]
	//
	// Blocking in (j, l, m) is analogous to Stage 1. The layouts of C
	// and D were chosen so that both are contiguous in m.
	// -----------------------------------------------------------------
	traverser(F, C, D)
		.order(
			into_blocks_dynamic<'j', 'J', 'j', 's'>(noarr::lit<tile_j>) ^
			into_blocks_dynamic<'l', 'L', 'l', 't'>(noarr::lit<tile_l>) ^
			into_blocks_dynamic<'m', 'M', 'm', 'u'>(noarr::lit<tile_m>) ^
			hoist<'J'>() ^ hoist<'L'>() ^ hoist<'M'>()
		)
		.template for_dims<'J', 'L'>([=](auto trav_JL) {
			trav_JL.template for_dims<'j', 's', 'l', 't'>([=](auto trav_jl) {
				num_t acc = num_t(0);

				trav_jl.template for_dims<'M', 'm', 'u'>([&](auto trav_jlm) {
					acc += C[trav_jlm] * D[trav_jlm];
				});

				F[trav_jl] = acc;
			});
		});

	// -----------------------------------------------------------------
	// Stage 3: G = E * F
	//   G[i, l] = sum_j E[i, j] * F[j, l]
	//
	// We again use 3D tiling, this time over (i, l, j). Layouts of E
	// and F were chosen so that both are contiguous in j, the reduction
	// dimension of this stage.
	// -----------------------------------------------------------------
	traverser(G, E, F)
		.order(
			into_blocks_dynamic<'i', 'I', 'i', 's'>(noarr::lit<tile_i>) ^
			into_blocks_dynamic<'l', 'L', 'l', 't'>(noarr::lit<tile_l>) ^
			into_blocks_dynamic<'j', 'J', 'j', 'u'>(noarr::lit<tile_j>) ^
			hoist<'I'>() ^ hoist<'L'>() ^ hoist<'J'>()
		)
		.template for_dims<'I', 'L'>([=](auto trav_IL) {
			trav_IL.template for_dims<'i', 's', 'l', 't'>([=](auto trav_il) {
				num_t acc = num_t(0);

				trav_il.template for_dims<'J', 'j', 'u'>([&](auto trav_ilj) {
					acc += E[trav_ilj] * F[trav_ilj];
				});

				G[trav_il] = acc;
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