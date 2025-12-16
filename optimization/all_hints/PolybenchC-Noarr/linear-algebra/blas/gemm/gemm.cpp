#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

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
	// C: i x j  (row-major: 'i' outer, 'j' inner)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k  (row-major: 'i' outer, 'k' inner)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	// B: k x j  (row-major: 'k' outer, 'j' inner)
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);
} tuning;

// -----------------------------------------------------------------------------
// Array initialization
// -----------------------------------------------------------------------------
void init_array(num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// Precompute lengths (used in formulas below)
	const std::size_t ni = C | get_length<'i'>();
	const std::size_t nj = B | get_length<'j'>();
	const std::size_t nk = A | get_length<'k'>();

	// Replace divisions inside loops with multiplication by precomputed inverses
	const num_t inv_ni = num_t(1) / num_t(ni);
	const num_t inv_nj = num_t(1) / num_t(nj);
	const num_t inv_nk = num_t(1) / num_t(nk);

	// C[i][j] = (i*j+1) % ni / ni
	{
		auto travC = traverser(C);

		// Parallelize across rows if OpenMP is enabled
		// (each iteration works on a distinct 'i' and is independent)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (auto trav_i : travC.template range<'i'>()) {
			trav_i.for_each([&](auto state) {
				auto [i, j] = get_indices<'i', 'j'>(state);
				C[state] = num_t((i * j + 1) % ni) * inv_ni;
			});
		}
	}

	// A[i][k] = i*(k+1) % nk / nk
	{
		auto travA = traverser(A);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (auto trav_i : travA.template range<'i'>()) {
			trav_i.for_each([&](auto state) {
				auto [i, k] = get_indices<'i', 'k'>(state);
				A[state] = num_t((i * (k + 1) % nk)) * inv_nk;
			});
		}
	}

	// B[k][j] = k*(j+2) % nj / nj
	{
		auto travB = traverser(B);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (auto trav_k : travB.template range<'k'>()) {
			trav_k.for_each([&](auto state) {
				auto [k, j] = get_indices<'k', 'j'>(state);
				B[state] = num_t((k * (j + 2) % nj)) * inv_nj;
			});
		}
	}
}

// -----------------------------------------------------------------------------
// Main computational kernel (GEMM)
//
// Form C := alpha*A*B + beta*C,
// A is NIxNK, B is NKxNJ, C is NIxNJ
//
// Layouts (row-major):
//   C[i][j] : dims 'i' (outer), 'j' (inner)
//   A[i][k] : dims 'i' (outer), 'k' (inner)
//   B[k][j] : dims 'k' (outer), 'j' (inner)
//
// Optimized strategy:
//   1. Scale C by beta:    C[i][j] = beta * C[i][j]
//   2. Accumulate product: C[i][j] += alpha * sum_k A[i][k] * B[k][j]
//      - Loop order: i -> k -> j
//        * For a fixed (i, k) we read A[i][k] once and reuse it across all j.
//        * Inner loop over j keeps accesses to B[k][j] and C[i][j] contiguous.
//      - Implemented entirely with Noarr traversers (including outer splits).
// -----------------------------------------------------------------------------
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

#pragma scop
	// -------------------------------------------------------------------------
	// 1. Scale C by beta: C[i][j] *= beta
	//    Traverse C alone; order (i outer, j inner) follows its layout.
	// -------------------------------------------------------------------------
	{
		auto travC = traverser(C);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (auto trav_i : travC.template range<'i'>()) {
			trav_i.for_each([&](auto state) {
				C[state] *= beta;
			});
		}
	}

	// -------------------------------------------------------------------------
	// 2. C += alpha * A * B
	//
	//    We use a joint traverser over (C, A, B) so that the state passed into
	//    lambdas can be used to index all three matrices.
	//
	//    Loop structure in terms of dimensions:
	//      for i:
	//        for k:
	//          let a_ik = A[i][k]
	//          for j:
	//            C[i][j] += alpha * a_ik * B[k][j]
	//
	//    This matches GEMM semantics while improving locality:
	//      - 'j' is the inner dimension for both B and C, so B[k][j] and
	//        C[i][j] are accessed in a cache-friendly, contiguous manner.
	//      - A[i][k] is loaded once per (i, k) and reused across the j-loop.
	// -------------------------------------------------------------------------
	{
		auto trav_all = traverser(C, A, B);

#ifdef _OPENMP
		// Parallelize across rows 'i'. Each iteration of this loop works on a
		// distinct value of 'i', so threads only write to disjoint rows of C
		// and read disjoint rows of A. B is read-only.
#pragma omp parallel for schedule(static)
#endif
		for (auto trav_i : trav_all.template range<'i'>()) {
			// trav_i: 'i' fixed, remaining dims are 'k' and 'j'

			// Outer loop over the shared dimension 'k'
			trav_i.template for_dims<'k'>([&](auto trav_ik) {
				// trav_ik: 'i' and 'k' fixed, remaining dim is 'j'

				// Load A[i][k] once and reuse it across all j
				const num_t a_ik = A[trav_ik];
				const num_t alpha_a_ik = alpha * a_ik;

				// Inner loop over 'j' (contiguous for both B and C)
				trav_ik.for_each([&](auto state) {
					// state has 'i', 'k', 'j'
					// C[i][j] += alpha * A[i][k] * B[k][j]
					C[state] += alpha_a_ik * B[state];
				});
			});
		}
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