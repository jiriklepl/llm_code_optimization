#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions (DATA_TYPE, N, M, DEFINE_PROTO_STRUCT, ...)
#include "defines.hpp"

// include benchmark-specific definitions (N, M, DATA_TYPE, etc.)
#include "syr2k.hpp"

using num_t = DATA_TYPE;

namespace {

// basic dimension proto-structures
constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

// layout tuning (can be changed independently of the algorithm)
struct tuning {
	// C is N x N, indexed by (i, j)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // physical layout: i (outer) x j (inner), row-major in (i,j)

	// A is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // physical layout: i (outer) x k (inner), row-major in (i,k)

	// B is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ i_vec); // physical layout: i (outer) x k (inner), row-major in (i,k)
} tuning;

// initialization function
// C: i x j  (N x N)
// A: i x k  (N x M)
// B: i x k  (N x M)
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	// Initialize A: A[i][k] = ( (i * k + 1) % n ) / n
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto n_len  = A | get_length<'i'>(); // corresponds to original 'n'
		A[state] = (num_t)((i * k + 1) % n_len) / n_len;
	});

	// Initialize B: B[i][k] = ( (i * k + 2) % m ) / m
	traverser(B).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto m_len  = B | get_length<'k'>(); // corresponds to original 'm'
		B[state] = (num_t)((i * k + 2) % m_len) / m_len;
	});

	// Initialize C: C[i][j] = ( (i * j + 3) % n ) / m
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto n_len  = C | get_length<'i'>(); // N
		auto m_len  = B | get_length<'k'>(); // M
		C[state] = (num_t)((i * j + 3) % n_len) / m_len;
	});
}

// computation kernel
// BLAS-like SYR2K:
//
//   C := alpha * A * B^T + alpha * B * A^T + beta * C
//
// with:
//   A: N x M
//   B: N x M
//   C: N x N (only lower-triangular part (i,j) with j <= i is updated)
//
// Optimized version notes:
//
// - We fuse the original two phases
//       C[i,j] *= beta;
//       C[i,j] += alpha * sum_k(...);
//   into a single pass per (i,j), loading C[i,j] once, scaling by beta,
//   accumulating the full reduction in a scalar, and writing C[i,j] once.
//   This greatly reduces memory traffic to C.
//
// - We still only update the lower triangle (j <= i); the upper triangle
//   remains unchanged as in the original code.
//
// - The reduction over k is expressed via a Noarr traverser and is
//   tiled by into_blocks_dynamic along 'k' to improve cache locality
//   without changing the visible (i,j,k) indices.
//
// - All loops are expressed with Noarr traversers / for_dims / for_each,
//   in accordance with the guidelines.
[[gnu::flatten, gnu::noinline]]
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// Tile size for the reduction dimension 'k'.
	// This is a moderate, architecture-agnostic choice; it can be tuned.
	constexpr std::size_t k_block = 64;

#pragma scop
	// Build a traverser over C, A, and B, and apply a blocking
	// transformation to the reduction dimension 'k':
	//   - into_blocks_dynamic<'k','K','k','t'>(k_block) splits 'k' into:
	//       * 'K' : block index
	//       *  'k': index within the block (re-using the original name)
	//       *  't': guard dimension for partial tail blocks
	//   - hoist<'K'>() moves the block index 'K' outward so that we
	//     process one k-block at a time.
	//
	// Crucially, when using order(...), the states passed to for_each /
	// for_dims still use the original dimension names ('i','j','k'),
	// so the rest of the kernel can be written in terms of the logical
	// problem indices without worrying about the internal blocking.
	auto trav = traverser(C, A, B).order(
		noarr::into_blocks_dynamic<'k', 'K', 'k', 't'>(noarr::lit<k_block>) ^
		noarr::hoist<'K'>()
	);

	// Outer loop over rows i of C.
	trav.template for_dims<'i'>([=](auto trav_i) {
		// Middle loop over columns j of C.
		// We keep the full [0, N) range and enforce the lower-triangular
		// domain with a simple j <= i check. This matches the original
		// semantics exactly (upper triangle is left untouched).
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			// Get the current (i, j) pair.
			auto state_ij = trav_ij.state();
			auto i_idx = get_index<'i'>(state_ij);
			auto j_idx = get_index<'j'>(state_ij);

			// Only update the lower triangle; j > i is ignored as before.
			if (j_idx > i_idx)
				return;

			// Load C[i,j] once, apply the beta scaling, and accumulate all
			// contributions for this pair into a local scalar.
			num_t c_ij = C[state_ij] * beta;

			// Reduction over the entire 'k' dimension.
			// Due to the blocking in 'k', this runs in a cache-friendly
			// order (K-tiles, then intra-tile k), but the state we see
			// here exposes only the original index 'k'.
			trav_ij.for_each([&](auto state_ijk) {
				// state_ijk carries indices (i, j, k).
				// For the symmetric rank-2k update, we need:
				//   - row i: A[i,k], B[i,k]
				//   - row j: A[j,k], B[j,k]

				// Row i is already encoded by 'i' in state_ijk.
				num_t A_ik = A[state_ijk];
				num_t B_ik = B[state_ijk];

				// Build a state where the 'i' index is replaced by j,
				// keeping the same k. Other indices in the state do not
				// matter for A/B, as they ignore unknown dimensions.
				auto state_jk = state_ijk.template with<index_in<'i'>>(j_idx);
				num_t A_jk = A[state_jk];
				num_t B_jk = B[state_jk];

				// Accumulate:
				//   C[i,j] += alpha * (A[j,k] * B[i,k] + B[j,k] * A[i,k]);
				c_ij += alpha * (A_jk * B_ik + B_jk * A_ik);
			});

			// Store the final value of C[i,j] once.
			C[state_ij] = c_ij;
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t m = M;

	// scalar parameters
	num_t alpha;
	num_t beta;

	// set lengths for all relevant dimensions
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n) ^
		noarr::set_length<'k'>(m);

	// allocate bags for C, A, B with selected layouts and lengths
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syr2k(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (only when some argument is given, mimicking PolyBench's DCE guard)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'i' to make the output row-major in (i, j)
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}