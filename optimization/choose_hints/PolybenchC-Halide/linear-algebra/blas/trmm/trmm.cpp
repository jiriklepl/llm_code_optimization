#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "trmm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec); // A: i x k
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ i_vec); // B: i x j
} tuning;

// initialization function
void init_array(int m, int n, num_t &alpha, auto A, auto B) {
	using namespace noarr;

	alpha = (num_t)1.5;

	// Initialize A: lower triangular (j < i) with values; diagonal = 1
	traverser(A) | for_dims<'i'>([=](auto trav_i) {
		auto s_i = trav_i.state();
		auto i = get_index<'i'>(s_i);

		// set A[i][i] = 1
		A[idx<'i','k'>(i, i)] = (num_t)1.0;

		// set A[i][k] for k < i
		if(i > 0) {
			std::size_t len_i = A | get_length<'i'>();
			trav_i.order(noarr::slice<'k'>(0, i)).for_each([=](auto s) {
				auto k = get_index<'k'>(s);
				A[s] = (num_t)((i + k) % len_i) / (num_t)len_i;
			});
		}
	});

	// Initialize B: B[i][j] = ((n + (i - j)) % n) / n
	traverser(B) | [=](auto s) {
		auto [i, j] = get_indices<'i','j'>(s);
		std::size_t len_j = B | get_length<'j'>();
		B[s] = (num_t)((len_j + (i - j)) % len_j) / (num_t)len_j;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_trmm(num_t alpha, auto A, auto B) {
	using namespace noarr;

	// create a view of B where 'i' is renamed to 'k' to access B[k][j]
	auto B_kj = B ^ noarr::rename<'i','k'>();

	#pragma scop
	traverser(B, A) | for_dims<'i'>([=](auto trav_i) {
		trav_i.template for_dims<'j'>([=](auto trav_ij) {
			auto si = trav_ij.state();
			auto i = get_index<'i'>(si);

			std::size_t m_len = B | get_length<'i'>();
			std::size_t start = i + 1;
			if(start < m_len) {
				std::size_t len = m_len - start;

				// for k from i+1 to M-1:
				trav_ij.order(noarr::slice<'k'>(start, len)).for_each([=](auto s) {
					B[s] += A[s] * B_kj[s]; // B[i][j] += A[k][i] * B[k][j]
				});
			}

			// B[i][j] = alpha * B[i][j]
			auto s_ij = trav_ij.state();
			B[s_ij] = alpha * B[s_ij];
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

	auto set_lengths =
		noarr::set_length<'i'>(m) ^
		noarr::set_length<'j'>(n) ^
		noarr::set_length<'k'>(m);

	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths); // i x k (MxM)
	auto B = noarr::bag(noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths); // i x j (MxN)

	// initialize data
	init_array((int)m, (int)n, alpha, A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_trmm(alpha, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, B.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
