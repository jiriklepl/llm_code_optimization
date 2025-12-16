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
[[gnu::flatten, gnu::noinline]]
void kernel_syrk(num_t alpha, num_t beta, auto C, auto A) {
	// BLAS SYRK:
	//   C := alpha * A * A^T + beta * C
	// A is N x M (i x k)
	// C is N x N (i x j), lower triangle (j <= i) updated
	using namespace noarr;

	#pragma scop
	traverser(C, A).template for_dims<'i'>([=](auto trav_i) {
		// First: C[i][j] *= beta for j <= i
		trav_i.template for_each<'j'>([=](auto state) {
			auto [i, j] = get_indices<'i', 'j'>(state);
			if (j <= i) {
				C[state] *= beta;
			}
		});

		// Then: for each k, accumulate alpha * A[i][k] * A[j][k] into C[i][j] for j <= i
		trav_i.template for_dims<'k'>([=](auto trav_k) {
			trav_k.template for_each<'j'>([=](auto state) {
				auto [i, j, k] = get_indices<'i', 'j', 'k'>(state);

				if (j <= i) {
					// A[i][k] uses current state's i,k (j is ignored by A)
					auto a_ik = A[state];

					// Build a state with i replaced by j to access A[j][k]
					auto state_jk = noarr::update_index<'i'>(state, [=](auto) { return j; });
					auto a_jk = A[state_jk];

					C[state] += alpha * a_ik * a_jk;
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