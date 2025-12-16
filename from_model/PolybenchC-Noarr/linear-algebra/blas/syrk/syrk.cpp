#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "syrk.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// C: i x j  (logical), layout defined by this proto-structure
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k  (logical), layout defined by this proto-structure
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A) {
	// C: i x j, length(i) = N, length(j) = N
	// A: i x k, length(i) = N, length(k) = M
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// A[i][k] = ( (i * k + 1) % n ) / n;
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		auto n_len = A | get_length<'i'>();
		A[state] = (num_t)((i * k + 1) % n_len) / n_len;
	});

	// C[i][j] = ( (i * j + 2) % m ) / m;
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto m_len = A | get_length<'k'>(); // use A's second dimension length (M)
		C[state] = (num_t)((i * j + 2) % m_len) / m_len;
	});
}

// computation kernel
//
// BLAS SYRK (lower triangle):
//   C := alpha * A * A^T + beta * C
// A is N x M (i x k)
// C is N x N (i x j), only the lower triangle (j <= i) is updated
//
// Optimizations implemented:
//
// 1. We keep the two logical phases (scale by beta, then accumulate alpha*A*A^T)
//    to preserve the original mathematical structure; we only reorder loops
//    and block the reduction dimension k.
//
// 2. We tile the k-dimension with noarr::into_blocks_dynamic<'k','K','k','u'>()
//    to improve cache locality when streaming A[i,k] / A[j,k]. The auxiliary
//    dimension 'u' acts as a guard for partial tiles so that we never touch
//    out-of-bounds elements.
//
// 3. For each fixed (i, k) inside a k-tile, we preload
//        a_ik = alpha * A[i,k]
//    once and reuse it across all j in that row. This reduces redundant
//    multiplications and shares the loaded A[i,k] value across the j-loop.
//
// 4. All loop nests are expressed using Noarr traversers and for_dims/for_each,
//    as required.
//
[[gnu::flatten, gnu::noinline]]
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
	using namespace noarr;

	// Tile size for the reduction dimension k. This can be tuned for a given
	// machine; 64 is a reasonable default on modern x64 CPUs.
	// Using a static value (noarr::lit<...>) lets the library optimize more.
	constexpr auto tile_k = noarr::lit<64>;

	#pragma scop

	// ---------------------------------------------------------------------
	// Phase 1: scale the lower triangle of C by beta
	// ---------------------------------------------------------------------
	// We traverse C alone so that we only visit each (i,j) once.
	traverser(C).for_each([&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		if (j <= i) {
			C[state] *= beta;
		}
	});

	// ---------------------------------------------------------------------
	// Phase 2: accumulate alpha * A * A^T into the lower triangle of C
	// ---------------------------------------------------------------------
	//
	// We use a multi-structure traverser over (C, A) and block the k-dimension:
	//   - into_blocks_dynamic<'k','K','k','u'>(tile_k) splits k into:
	//       K ... block index
	//       k ... index within the block
	//       u ... guard dimension (length 0/1) for incomplete blocks
	//   - We iterate as:
	//       for i
	//         for each K
	//           for each (k,u) within this block
	//             a_ik = alpha * A[i,k]
	//             for each j
	//               if (j <= i) C[i,j] += a_ik * A[j,k]
	//
	// This preserves the original computation:
	//   C[i,j] = beta * C[i,j] + alpha * sum_k A[i,k] * A[j,k]
	// for all j <= i, and leaves j > i unchanged.
	traverser(C, A)
		.order(into_blocks_dynamic<'k', 'K', 'k', 'u'>(tile_k))
		// Outer loop over i (rows of C / A)
		.template for_dims<'i'>([&](auto trav_i) {
			// For each block K of the reduction dimension k
			trav_i.template for_dims<'K'>([&](auto trav_iK) {
				// Iterate over valid (k,u) pairs inside this K-block.
				// Using into_blocks_dynamic ensures that for out-of-range
				// (K,k) combinations, the length of 'u' is zero, and this
				// for_dims<'k','u'> therefore skips them entirely.
				trav_iK.template for_dims<'k', 'u'>([&](auto trav_ik) {
					// trav_ik has i, K, k, u fixed; remaining dimension is j.
					// Compute alpha * A[i,k] once and reuse over j.
					auto state_ik = trav_ik.state();
					num_t a_ik = alpha * A[state_ik];

					// Now iterate over all columns j for this fixed (i,k).
					trav_ik.for_each([&](auto state) {
						// state carries i, j, k (and K,u) indices.
						auto [i, j] = get_indices<'i', 'j'>(state);

						// Only update the lower triangle, as in the original code.
						if (j <= i) {
							// To get A[j,k], reuse the current state but replace i by j.
							auto state_jk = noarr::update_index<'i'>(state, [=](auto) { return j; });
							num_t a_jk = A[state_jk];

							C[state] += a_ik * a_jk;
						}
					});
				});
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

	// input data
	num_t alpha;
	num_t beta;

	// set lengths for each structure
	auto set_lengths_C = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);
	auto set_lengths_A = noarr::set_length<'i'>(n) ^ noarr::set_length<'k'>(m);

	// allocate bags
	auto C = noarr::bag(noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths_C);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths_A);

	// initialize data
	init_array(alpha, beta, C.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_syrk(alpha, beta, C.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (optional)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		// hoist 'i' to print C in i-major order
		noarr::serialize_data(std::cerr, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}