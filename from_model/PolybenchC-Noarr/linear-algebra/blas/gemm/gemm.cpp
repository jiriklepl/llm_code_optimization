#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // C: i x j   (row-major in j)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // A: i x k   (row-major in k)
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec); // B: k x j   (row-major in j)
} tuning;

// Array initialization
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// C[i][j] = (i*j+1) % ni / ni
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto ni = C | get_length<'i'>();
		C[state] = (num_t)((i * j + 1) % ni) / ni;
	});

	// A[i][k] = i*(k+1) % nk / nk
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto nk = A | get_length<'k'>();
		A[state] = (num_t)(i * (k + 1) % nk) / nk;
	});

	// B[k][j] = k*(j+2) % nj / nj
	traverser(B).for_each([=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		auto nj = B | get_length<'j'>();
		B[state] = (num_t)(k * (j + 2) % nj) / nj;
	});
}

// Main computational kernel (GEMM)
// Form C := alpha*A*B + beta*C,
// A is NIxNK, B is NKxNJ, C is NIxNJ
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	// Problem sizes (queried from layouts, not from globals)
	const std::size_t ni = C | get_length<'i'>();
	const std::size_t nj = C | get_length<'j'>();
	const std::size_t nk = A | get_length<'k'>();

	// Simple cache/blocking parameters.
	// These are chosen to be small enough to fit typical L1/L2 caches while
	// still amortizing loop overhead; they can be tuned further if needed.
	constexpr std::size_t tile_i = 64;
	constexpr std::size_t tile_j = 64;
	constexpr std::size_t tile_k = 32;

#pragma scop
	// If the dimensions are not multiples of the tile sizes, fall back to the
	// original, unblocked kernel. This guarantees correctness for all sizes,
	// and the blocked kernel is used whenever it is safe.
	if (ni < tile_i || nj < tile_j || nk < tile_k ||
	    ni % tile_i != 0 || nj % tile_j != 0 || nk % tile_k != 0) {

		// Original traversal structure, kept for full generality.
		// Loops:
		//   for i:
		//     for j: C[i,j] *= beta;
		//     for k, j: C[i,j] += alpha * A[i,k] * B[k,j];
		traverser(C, A, B).template for_dims<'i'>([=](auto inner) {
			// First: scale C by beta along j for the current row i.
			inner.template for_each<'j'>([=](auto state) {
				C[state] *= beta;
			});

			// Then: accumulate alpha * A[i,k] * B[k,j] into C[i,j].
			inner.for_each([=](auto state) {
				C[state] += alpha * A[state] * B[state];
			});
		});
	} else {
		// -------------------------
		// Blocked / tiled kernel
		// -------------------------
		//
		// Layout recap:
		//   C: dims (i, j), j contiguous
		//   A: dims (i, k), k contiguous
		//   B: dims (k, j), j contiguous
		//
		// After applying into_blocks on 'i', 'j', 'k', we conceptually obtain:
		//   i -> I (block index) and i (in-block index)
		//   j -> J (block index) and j (in-block index)
		//   k -> K (block index) and k (in-block index)
		//
		// We then iterate tiles of C indexed by (I, J). For each tile:
		//   1) Scale C-entries in the tile by beta (once per (i,j)).
		//   2) Sweep over K-tiles; in each K-tile:
		//        for each (i,k) in the tile:
		//          a = A[i,k]
		//          for each j in the J-tile:
		//            C[i,j] += alpha * a * B[k,j]
		//
		// This improves spatial locality for A and B and increases cache reuse,
		// while preserving the original mathematical semantics.

		// Proto-structure that describes 3D blocking of (i, j, k).
		auto blocking_proto =
			into_blocks<'i', 'I', 'i'>(tile_i) ^
			into_blocks<'j', 'J', 'j'>(tile_j) ^
			into_blocks<'k', 'K', 'k'>(tile_k);

		// Traverse blocked views of C, A, and B together.
		traverser(C, A, B)
			.order(blocking_proto)
			// Outer loops over tiles of C: I -> rows, J -> columns.
			.template for_dims<'I', 'J'>([=](auto trav_IJ) {
				// ----------------------------------
				// 1) Scale C in this (I,J) tile
				// ----------------------------------
				// We loop only over in-tile (i,j). Dimensions K,k are left
				// unfixed here and are ignored when indexing C (C has no k-dim).
				trav_IJ.template for_dims<'i', 'j'>([=](auto trav_ij) {
					C[trav_ij] *= beta;
				});

				// ----------------------------------
				// 2) Accumulate alpha * A * B into C
				// ----------------------------------
				// Sweep k in blocks K to improve cache locality.
				trav_IJ.template for_dims<'K'>([=](auto trav_IJK) {
					// For the current K-block and (I,J) tile, traverse all
					// in-tile (i,k) pairs. For each such pair we:
					//   - load A[i,k] once and scale it by alpha,
					//   - update the entire row of C[i, *] over j in this tile.
					trav_IJK.template for_dims<'i', 'k'>([=](auto trav_ik) {
						// A and B share dims (i,k) / (k, j); for the current
						// (i,k) within a tile, A[trav_ik] is reused across
						// all j in the J-tile.
						num_t a_ik = A[trav_ik];
						num_t scaled = alpha * a_ik;

						// Only j remains free; for_each will iterate j over
						// the current J-tile. The same state can be used to
						// index C and B because they share dims (k,j) / (i,j)
						// as appropriate; extra dims in the state are ignored
						// by structures that don't use them.
						trav_ik.for_each([=](auto state) {
							C[state] += scaled * B[state];
						});
					});
				});
			});
	}
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;

	// scalars
	num_t alpha;
	num_t beta;

	// lengths proto-structure (applied to all matrices)
	auto set_lengths =
		noarr::set_length<'i'>(ni)
		^ noarr::set_length<'j'>(nj)
		^ noarr::set_length<'k'>(nk);

	// bags for C, A, B with appropriate layouts
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths); // i x j
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths); // i x k
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths); // k x j

	// initialize arrays
	init_array(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// start timer
	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	// stop timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// output to prevent dead-code elimination (similar role as print_array/polybench_prevent_dce)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}