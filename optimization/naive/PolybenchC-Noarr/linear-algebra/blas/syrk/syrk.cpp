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
	// C: i x j  (logical), stored as j (outer) x i (inner)
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	// A: i x k  (logical), stored as k (outer) x i (inner)
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
} tuning;

// initialization function
void init_array(num_t &alpha, num_t &beta, auto C, auto A) {
	// C: i x j, length(i) = N, length(j) = N
	// A: i x k, length(i) = N, length(k) = M
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	// Precompute lengths once (avoid querying get_length for every element)
	const auto n_len = A | get_length<'i'>();
	const auto m_len = A | get_length<'k'>();

	// A[i][k] = ( (i * k + 1) % n ) / n;
	traverser(A).for_each([&](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)((i * k + 1) % n_len) / n_len;
	});

	// C[i][j] = ( (i * j + 2) % m ) / m;
	traverser(C).for_each([&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 2) % m_len) / m_len;
	});
}

// computation kernel
// C := alpha * A * A^T + beta * C
// A is N x M (i x k)
// C is N x N (i x j), lower triangle (j <= i) updated
[[gnu::flatten, gnu::noinline]]
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
	using namespace noarr;

	#pragma scop
	// Traverse over shared dimension 'i' first (one row/column of C at a time)
	traverser(C, A).template for_dims<'i'>([&](auto trav_i) {
		// Fix the current i once and reuse it in inner lambdas
		const auto state_i = trav_i.state();
		const auto i = get_index<'i'>(state_i);

		// --------------------------------------------------------------------
		// 1) Scale C[i][j] by beta for j <= i
		// --------------------------------------------------------------------
		//
		// We iterate only over dimension 'j'. The traverser passed to the
		// lambda has both 'i' and 'j' fixed and no loops over 'k' are
		// performed here. This keeps this phase strictly 2D over C.
		trav_i.template for_dims<'j'>([&](auto trav_ij) {
			auto state_ij = trav_ij.state();
			auto j = get_index<'j'>(state_ij);

			if (j <= i) {
				// C uses only dimensions 'i' and 'j'; any 'k' present in
				// the combined traverser is ignored by C.
				C[trav_ij] *= beta;
			}
		});

		// --------------------------------------------------------------------
		// 2) Rank-k update: C[i][j] += alpha * A[i][k] * A[j][k]  for j <= i
		// --------------------------------------------------------------------
		//
		// Loop order: i (outer) -> k (middle) -> j (inner)
		//
		// For each (i,k) pair we:
		//   - load A[i][k] once and reuse it across all j in the inner loop
		//   - iterate over j and load A[j][k] for each j
		//
		// This hoists A[i][k] out of the j-loop, reducing redundant loads
		// and improving temporal locality.
		trav_i.template for_dims<'k'>([&](auto trav_ik) {
			// State with 'i' and 'k' fixed (no 'j' yet)
			auto state_ik = trav_ik.state();
			const auto k = get_index<'k'>(state_ik);

			// Pre-load A[i][k] once for this (i,k); A ignores 'j'
			const num_t a_ik = A[state_ik];

			// Now iterate over j, still within this fixed (i,k)
			trav_ik.template for_dims<'j'>([&](auto trav_ijk) {
				auto state_ijk = trav_ijk.state();
				auto j = get_index<'j'>(state_ijk);

				if (j <= i) {
					// A[i][k] is already in a_ik.
					//
					// For A[j][k] we need a state whose 'i' index is 'j'
					// instead of the current row index 'i'. We construct it
					// efficiently via update_index on the current state.
					auto state_jk = update_index<'i'>(state_ijk, [j](auto) { return j; });
					const num_t a_jk = A[state_jk];

					// C[i][j] += alpha * A[i][k] * A[j][k]
					C[trav_ijk] += alpha * a_ik * a_jk;
				}
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