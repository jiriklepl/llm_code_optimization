#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "trmm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// A: i x k  (M x M), logical indexing A[i][k]
	// Layout: k_vec ^ i_vec -> scalar ^ k ^ i  == row-major in k for fixed i
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);

	// B: i x j  (M x N), logical indexing B[i][j]
	// Layout: j_vec ^ i_vec -> scalar ^ j ^ i  == row-major in j for fixed i
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec);
} tuning;

// Array initialization
void init_array(num_t &alpha, auto A, auto B) {
	using namespace noarr;

	// A: i x k, square M x M
	// B: i x j, M x N

	alpha = (num_t)1.5;

	const int m_len = static_cast<int>(A | get_length<'i'>());
	const int n_len = static_cast<int>(B | get_length<'j'>());

	// Initialize A: unit lower-triangular
	// for (i = 0; i < m; i++) {
	//   for (j = 0; j < i; j++) A[i][j] = ((i + j) % m) / m;
	//   A[i][i] = 1.0;
	// }
	traverser(A).for_each([=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		int ii = static_cast<int>(i);
		int kk = static_cast<int>(k);

		if (kk < ii) {
			A[state] = (num_t)((ii + kk) % m_len) / (num_t)m_len;
		} else if (kk == ii) {
			A[state] = (num_t)1.0;
		}
		// For kk > ii we leave A[state] uninitialized, matching the C code
	});

	// Initialize B
	// for (i = 0; i < m; i++)
	//   for (j = 0; j < n; j++)
	//     B[i][j] = ((n + (i - j)) % n) / n;
	traverser(B).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		int ii = static_cast<int>(i);
		int jj = static_cast<int>(j);

		B[state] = (num_t)(((n_len + (ii - jj)) % n_len)) / (num_t)n_len;
	});
}

// Main computational kernel
// Original C (PolyBench TRMM kernel):
//
// for (i = 0; i < _PB_M; i++)
//   for (j = 0; j < _PB_N; j++) {
//     for (k = i+1; k < _PB_M; k++)
//       B[i][j] += A[k][i] * B[k][j];
//     B[i][j] = alpha * B[i][j];
//   }
//
// A is MxM (i x k), B is MxN (i x j)
//
// Optimized formulation used here (per row i):
//
//   // Triangular update written in "row-update" form:
//   for (k = i+1; k < M; k++) {
//     let a_ki = A[k][i];
//     for (j = 0; j < N; j++)
//       B[i][j] += a_ki * B[k][j];
//   }
//   // Final scaling of the whole row i
//   for (j = 0; j < N; j++)
//     B[i][j] *= alpha;
//
// This is algebraically equivalent:
//   - For fixed (i,j) the sum over k is unchanged (we only swapped j and k).
//   - Different j are independent, so reordering j vs. k is safe.
//   - The i-loop remains outermost and increasing, preserving the requirement
//     that B[k][j] (for k > i) is read before B[k][j] is updated when row k
//     itself is processed.
//
// Implementation details:
//   * We keep 'i' as the outer loop using a Noarr traverser over B.
//   * For each row i, we view B[i,*] and B[k,*] as 1D structures in 'j'.
//   * For each pair (i,k) with k > i we perform a SAXPY-like update
//       B[i,*] += A[k,i] * B[k,*]
//     using a Noarr traverser over the row dimension 'j'.
//   * After finishing all k for a given row i, we scale B[i,*] by alpha
//     in another traverser over 'j'.
//
// Compared to the original kernel:
//   - A[k,i] is now loaded once per (i,k) and reused across all N columns.
//   - B rows B[i,*] and B[k,*] are traversed contiguously in 'j', which
//     greatly improves spatial locality and enables vectorization of the
//     inner SAXPY over 'j'.
[[gnu::flatten, gnu::noinline]]
void kernel_trmm(num_t alpha, auto A, auto B) {
	using namespace noarr;

	const std::size_t m_len = A | get_length<'i'>();

	#pragma scop
	// Outer loop over rows i. This must stay sequential to preserve
	// the dependence B[i,j] uses on B[k,j] for k > i.
	traverser(B).template for_dims<'i'>([&](auto trav_i) {
		// trav_i has 'i' fixed, and will be used to build 1D row views.
		auto s_i = trav_i.state();
		const std::size_t i = get_index<'i'>(s_i);

		// 1D view of the destination row B[i,*] over dimension 'j'.
		// Layout: scalar<num_t> ^ j_vec, contiguous in 'j'.
		auto B_row_i = B ^ noarr::fix<'i'>(i);

		// Triangular update:
		//   for (k = i+1; k < M; k++)
		//     B[i,*] += A[k,i] * B[k,*]
		for (std::size_t k = i + 1; k < m_len; ++k) {
			// A[k][i] is stored as A at indices (i = k, k = i).
			auto a_state = noarr::idx<'i', 'k'>(k, i);
			const num_t a_ki = A[a_state];

			// Optional tiny optimization: skip clearly zero multipliers
			// (keeps semantics, just avoids useless SAXPYs if initialization
			// ever produces zeros on the strict lower part).
			if (a_ki == num_t(0))
				continue;

			// 1D view of the source row B[k,*] over 'j'.
			auto B_row_k = B ^ noarr::fix<'i'>(k);

			// SAXPY over the whole row:
			//   for all j: B[i,j] += a_ki * B[k,j]
			noarr::traverser(B_row_i, B_row_k).for_each([&](auto s_j) {
				B_row_i[s_j] += a_ki * B_row_k[s_j];
			});
		}

		// Final scaling of the whole row i:
		//   for (j = 0; j < N; j++) B[i][j] *= alpha;
		noarr::traverser(B_row_i).for_each([&](auto s_j) {
			B_row_i[s_j] *= alpha;
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t m = M;
	std::size_t n = N;

	// input data
	num_t alpha;

	// set lengths for all dimensions used
	auto set_lengths = noarr::set_length<'i'>(m)
	                 ^ noarr::set_length<'j'>(n)
	                 ^ noarr::set_length<'k'>(m);

	// allocate bags
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths); // M x M
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths); // M x N

	// initialize data
	init_array(alpha, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trmm(alpha, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (for DCE prevention / correctness check)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, B.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}