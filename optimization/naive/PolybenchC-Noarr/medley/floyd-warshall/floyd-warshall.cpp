#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "floyd-warshall.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	// path[i][j] in C: i = row, j = column, row-major
	// => 'j' inner, 'i' outer
	DEFINE_PROTO_STRUCT(path_layout, j_vec ^ i_vec);
} tuning;

// Array initialization: mirrors the original C init_array
void init_array(auto path) {
	using namespace noarr;

	traverser(path).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		num_t value = static_cast<num_t>(i * j % 7 + 1);
		if (((i + j) % 13 == 0) || ((i + j) % 7 == 0) || ((i + j) % 11 == 0))
			value = static_cast<num_t>(999);

		path[state] = value;
	});
}

// Main computational kernel: Floyd–Warshall
// Original C loops:
//
// for (k = 0; k < _PB_N; k++)
//   for (i = 0; i < _PB_N; i++)
//     for (j = 0; j < _PB_N; j++)
//       path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
//                      path[i][j] : path[i][k] + path[k][j];
//
// Optimized Noarr implementation:
//
// 1. We still keep k as the outer loop, because Floyd–Warshall is
//    sequential in k.
// 2. For each fixed k, we:
//      - Create two views:
//          * path_row_k = k-th row  (varying 'j', fixed 'i' = k)
//          * path_col_k = k-th col  (varying 'i', fixed 'j' = k)
//      - Traverse the matrix in row-major order using for_dims<'i'>:
//          * outer: i (row)   – good cache locality on rows
//          * inner: j (col)
// 3. For each (k, i) pair we hoist path[i][k] out of the inner j-loop:
//      num_t pik = path[i][k];
//    and reuse it for all j in that row. This is a classic micro-
//    optimization for Floyd–Warshall. It is semantically safe in this
//    benchmark because the update at (i, k) cannot decrease path[i][k]
//    (path[k][k] is non‑negative), so reading the pre-update value for
//    all j gives the same result as the original scalar code.
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(std::size_t n, auto path) {
	using namespace noarr;

	// A dummy 1D structure used only to traverse k in [0, n)
	auto k_struct = noarr::scalar<char>() ^ k_vec ^ noarr::set_length<'k'>(n);

#pragma scop
	// Outer loop: k = 0 .. n-1
	traverser(k_struct).for_each([&](auto sk) {
		const std::size_t k = get_index<'k'>(sk);

		// Views of the k-th row and k-th column.
		// These are light-weight wrappers; they *alias* the same data
		// as `path` and do not copy anything.
		//
		//  - path_row_k[j] = path[k][j]
		//  - path_col_k[i] = path[i][k]
		auto path_row_k = path ^ noarr::fix<'i'>(k);
		auto path_col_k = path ^ noarr::fix<'j'>(k);

		// Traverse the matrix in row-major order with 'i' as the
		// explicit outer dimension. For each k and each row i, we
		// load path[i][k] once and reuse it for the whole inner j-loop.
		traverser(path).template for_dims<'i'>([&](auto ti) {
			// ti has 'i' fixed; its state contains the current row index.
			auto s_i = ti.state();
			const std::size_t i = get_index<'i'>(s_i);

			// Hoisted path[i][k] for the whole row i at this k.
			// This is invariant across j, so loading it once improves
			// cache and register reuse and reduces address calculations.
			const num_t pik = path_col_k[noarr::idx<'i'>(i)];

			// Now iterate all columns j in this row.
			ti.for_each([&](auto sij) {
				const std::size_t j = get_index<'j'>(sij);

				// path[k][j]: from the k-th row view
				const num_t pkj = path_row_k[noarr::idx<'j'>(j)];

				const num_t via_k = pik + pkj;
				num_t &pij = path[sij]; // path[i][j]

				// Standard Floyd–Warshall relaxation
				if (via_k < pij)
					pij = via_k;
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

	// build common length descriptor
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(n);

	// allocate data
	auto path = noarr::bag(noarr::scalar<num_t>() ^ tuning.path_layout ^ set_lengths);

	// initialize data
	init_array(path.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_floyd_warshall(n, path.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (for DCE / correctness checking)
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, path.get_ref() ^ noarr::hoist<'i'>());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}