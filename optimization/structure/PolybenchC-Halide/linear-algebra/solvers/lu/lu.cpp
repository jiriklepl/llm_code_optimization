#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "lu.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(int n, auto A) {
	using namespace noarr;

	// A is n x n, layout: i x j (row i, column j)

	// initialize lower triangle (including diagonal)
	traverser(A) | for_dims<'i'>([=](auto trav_i) {
		auto si = trav_i.state();
		std::size_t i = get_index<'i'>(si);

		// j in [0, i]  -> A[i][j] = (-j % n)/n + 1
		trav_i.order(noarr::slice<'j'>(0, i + 1)) | [=](auto s) {
			std::size_t j = get_index<'j'>(s);
			int temp = (-(int)j) % n;
			num_t val = (num_t)temp / (num_t)n + (num_t)1;
			A[s] = val;
		};

		// j in [i+1, n) -> A[i][j] = 0
		if (i + 1 <= (std::size_t)n) {
			trav_i.order(noarr::slice<'j'>(i + 1, (std::size_t)n - (i + 1))) | [=](auto s) {
				A[s] = (num_t)0;
			};
		}

		// A[i][i] = 1
		A[noarr::idx<'i','j'>(i, i)] = (num_t)1;
	});

	// Make the matrix positive semi-definite (like in cholesky)
	auto B = noarr::bag(A.structure());

	auto B_ref = B.get_ref();

	// zero B
	traverser(B) | [=](auto s) {
		B_ref[s] = (num_t)0;
	};

	// B[r][s] += A[r][t] * A[s][t]  for all t
	traverser(B_ref ^ noarr::bcast<'t'>((std::size_t)n)) | for_dims<'t'>([=](auto trav_t) {
		trav_t | for_dims<'i'>([=](auto trav_r) {
			trav_r | for_each<'j'>([=](auto state) {
				auto [r, s, t] = get_indices<'i','j','t'>(state);
				B_ref[state] += A[noarr::idx<'i','j'>(r, t)] * A[noarr::idx<'i','j'>(s, t)];
			});
		});
	});

	// copy B back to A
	traverser(A, B) | [=](auto s) {
		A[s] = B_ref[s];
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_lu(auto A) {
	using namespace noarr;

	const std::size_t n = A | get_length<'i'>();

	traverser(A) | for_dims<'i'>([=](auto inner) {
		auto si = inner.state();
		std::size_t i = get_index<'i'>(si);

		// for (j = 0; j < i; j++)
		inner.order(noarr::slice<'j'>(0, i)) | for_dims<'j'>([=](auto trav_j) {
			auto sj = trav_j.state();
			std::size_t j = get_index<'j'>(sj);

			// for (k = 0; k < j; k++) A[i][j] -= A[i][k] * A[k][j];
			traverser(noarr::scalar<void>() ^ noarr::bcast<'k'>(j)) | for_dims<'k'>([=](auto trav_k) {
				auto sk = sj & trav_k.state();
				std::size_t k = get_index<'k'>(sk);
				std::size_t ii = get_index<'i'>(sk);
				std::size_t jj = get_index<'j'>(sk);

				A[noarr::idx<'i','j'>(ii, jj)] -= A[noarr::idx<'i','j'>(ii, k)] * A[noarr::idx<'i','j'>(k, jj)];
			});

			// A[i][j] /= A[j][j];
			A[noarr::idx<'i','j'>(i, j)] /= A[noarr::idx<'i','j'>(j, j)];
		});

		// for (j = i; j < n; j++)
		inner.order(noarr::slice<'j'>(i, n - i)) | for_dims<'j'>([=](auto trav_j) {
			auto sj = trav_j.state();
			std::size_t j = get_index<'j'>(sj);

			// for (k = 0; k < i; k++) A[i][j] -= A[i][k] * A[k][j];
			traverser(noarr::scalar<void>() ^ noarr::bcast<'k'>(i)) | for_dims<'k'>([=](auto trav_k) {
				auto sk = sj & trav_k.state();
				std::size_t k = get_index<'k'>(sk);
				std::size_t ii = get_index<'i'>(sk);
				std::size_t jj = get_index<'j'>(sk);

				A[noarr::idx<'i','j'>(ii, jj)] -= A[noarr::idx<'i','j'>(ii, k)] * A[noarr::idx<'i','j'>(k, jj)];
			});
		});
	});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// structure and bag for A
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);

	// initialize data
	init_array((int)n, A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_lu(A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
