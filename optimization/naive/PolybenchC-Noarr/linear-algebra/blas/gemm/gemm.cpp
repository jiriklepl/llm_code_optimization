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
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec); // C: i x j (row-major: j inner, i outer)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // A: i x k (row-major: k inner, i outer)
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec); // B: k x j (row-major: j inner, k outer)
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
//
// Optimized for better data locality:
// 1) First scale C by beta in a separate pass (no dependence on A/B).
// 2) Then perform the multiplication-addition with explicit loop order i -> k -> j.
//    For a fixed (i, k) we load A[i, k] once and reuse it across the whole j-row,
//    which yields contiguous accesses to both B[k, j] and C[i, j].
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	using namespace noarr;

#pragma scop
	// ------------------------------------------------------------------
	// Step 1: C[i][j] *= beta
	// ------------------------------------------------------------------
	// Traverses C only; each element is scaled exactly once.
	traverser(C).for_each([=](auto state) {
		C[state] *= beta;
	});

	// ------------------------------------------------------------------
	// Step 2: C[i][j] += alpha * sum_k A[i][k] * B[k][j]
	// ------------------------------------------------------------------
	// We traverse the three matrices together and explicitly choose the
	// loop-nest order:
	//   for i
	//     for k
	//       for j
	//
	// - Outer 'i': selects the row of A and C.
	// - Middle 'k': walks along the row of A; we load A[i,k] once.
	// - Inner 'j': walks along rows of B and C, which are contiguous
	//              in memory thanks to the chosen layouts (j is inner).
	traverser(C, A, B)
		.template for_dims<'i'>([=](auto trav_i) {
			// trav_i: 'i' fixed, 'k' and 'j' remain

			// for k ...
			trav_i.template for_dims<'k'>([=](auto trav_ik) {
				// trav_ik: 'i' and 'k' fixed, 'j' remains

				// State with indices (i, k); used for A[i, k].
				auto state_ik = trav_ik.state();

				// Load A[i,k] once, scale it by alpha and reuse for all j.
				num_t a_ik = alpha * A[state_ik];

				// for j ...
				// Only 'j' is unfixed in trav_ik, so this iterates j-contiguously.
				trav_ik.for_each([=](auto state) {
					// state: (i, j, k)
					// C uses (i, j); B uses (k, j); A already loaded as a_ik.
					C[state] += a_ik * B[state];
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