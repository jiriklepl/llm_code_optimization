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
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // physical layout: j (outer) x i (inner)

	// A is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // physical layout: k (outer) x i (inner)

	// B is N x M, indexed by (i, k)
	DEFINE_PROTO_STRUCT(b_layout, k_vec ^ i_vec); // physical layout: k (outer) x i (inner)
} tuning;

// initialization function
// C: i x j  (N x N)
// A: i x k  (N x M)
// B: i x k  (N x M)
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta  = (num_t)1.2;

	// Pre-query lengths once; they are constant for the whole run
	const auto n_len_A = A | get_length<'i'>(); // N, for A
	const auto m_len_B = B | get_length<'k'>(); // M, for B
	const auto n_len_C = C | get_length<'i'>(); // N, for C

	// Initialize A: A[i][k] = ( (i * k + 1) % n ) / n
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)((i * k + 1) % n_len_A) / n_len_A;
	});

	// Initialize B: B[i][k] = ( (i * k + 2) % m ) / m
	traverser(B).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		B[state] = (num_t)((i * k + 2) % m_len_B) / m_len_B;
	});

	// Initialize C: C[i][j] = ( (i * j + 3) % n ) / m
	traverser(C).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 3) % n_len_C) / m_len_B;
	});
}

// computation kernel
// BLAS-like SYR2K:
//
// C := alpha * A * B^T + alpha * B * A^T + beta * C
//
// with:
//   A: N x M
//   B: N x M
//   C: N x N (only lower-triangular part (i,j) with j <= i is updated)
[[gnu::flatten, gnu::noinline]]
void kernel_syr2k(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

	// C: i x j
	// A: i x k
	// B: i x k

	// Precompute lengths; these are used to build slices for triangular loops.
	const auto n_len = C | get_length<'i'>(); // N
	// k-length is the same on A and B
	const auto m_len = A | get_length<'k'>(); // M

#pragma scop
	// -------------------------------------------------------------------------
	// 1) Scale the lower triangular part of C by beta:
	//
	//    for (i = 0; i < N; i++)
	//      for (j = 0; j <= i; j++)
	//        C[i][j] *= beta;
	//
	// We reorganize this into a j-major traversal:
	//
	//    for (j = 0; j < N; j++)
	//      for (i = j; i < N; i++)
	//        C[i][j] *= beta;
	//
	// This order is more cache-friendly for the chosen layout of C
	// (j outer, i inner), while touching each (i,j) with j <= i exactly once.
	// No explicit j<=i condition is needed – it is encoded in the slice.
	// -------------------------------------------------------------------------
	traverser(C).template for_dims<'j'>([=](auto trav_j) {
		// 'j' is fixed in trav_j, 'i' is still free
		auto state_j = trav_j.state();
		auto j_idx   = get_index<'j'>(state_j);

		// We want i in [j_idx, n_len), i.e. length n_len - j_idx
		const auto len_i = n_len - j_idx;

		// Restrict the traverser to the lower-triangular part for this column j
		trav_j
			.order(noarr::slice<'i'>(j_idx, len_i))
			.for_each([=](auto state) {
				// state has indices (i, j) with i in [j, N)
				C[state] *= beta;
			});
	});

	// -------------------------------------------------------------------------
	// 2) Rank-2k update of the lower triangular part:
	//
	//    for (i = 0; i < N; i++)
	//      for (k = 0; k < M; k++)
	//        for (j = 0; j <= i; j++)
	//          C[i][j] += alpha * A[j][k] * B[i][k]
	//                   + alpha * B[j][k] * A[i][k];
	//
	// We reorganize the triple loop to:
	//
	//    for (k = 0; k < M; k++)
	//      for (j = 0; j < N; j++)
	//        (preload A[j][k], B[j][k])
	//        for (i = j; i < N; i++)
	//          C[i][j] += A[i][k] * (alpha * B[j][k])
	//                  + B[i][k] * (alpha * A[j][k]);
	//
	// For each fixed (k, j) we:
	//   - read A[j][k] and B[j][k] once,
	//   - traverse i as the innermost dimension, which is contiguous for
	//     all of C, A and B under the chosen layouts (i is the inner dim).
	//
	// Semantics are preserved:
	//   - each triple (i,j,k) with j <= i is visited exactly once,
	//   - for each (i,j), contributions over k are still accumulated in
	//     increasing k order, as in the original code.
	// -------------------------------------------------------------------------
	traverser(C, A, B).template for_dims<'k'>([=](auto trav_k) {
		// 'k' is fixed here; 'i' and 'j' are still available
		trav_k.template for_dims<'j'>([=](auto trav_kj) {
			// For this (k, j) pair, preload A[j][k] and B[j][k]
			auto state_kj = trav_kj.state();
			auto j_idx    = get_index<'j'>(state_kj);

			// Build a state with i = j (A and B have dims (i, k); j is
			// represented by the same 'i' dimension there).
			auto state_jk = state_kj.template with<index_in<'i'>>(j_idx);

			const num_t A_jk = A[state_jk];
			const num_t B_jk = B[state_jk];

			// Precompute alpha * A[j][k] and alpha * B[j][k], reused across i
			const num_t alpha_A_jk = alpha * A_jk;
			const num_t alpha_B_jk = alpha * B_jk;

			// Only update the lower triangular part: i in [j, N)
			const auto len_i = n_len - j_idx;

			trav_kj
				.order(noarr::slice<'i'>(j_idx, len_i))
				.for_each([=](auto state) {
					// state has indices (i, j, k) with i in [j, N)
					const num_t A_ik = A[state]; // A[i][k]
					const num_t B_ik = B[state]; // B[i][k]

					// C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
					//          = A_ik * (alpha * B_jk) + B_ik * (alpha * A_jk)
					C[state] += A_ik * alpha_B_jk + B_ik * alpha_A_jk;
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