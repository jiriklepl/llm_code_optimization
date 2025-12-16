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

	// Tunable blocking factor for the inner 'j' dimension in the
	// Floyd–Warshall kernel. This steers the width of tiles processed
	// in the innermost loop to improve cache use and vectorization.
	static constexpr std::size_t j_block_size = 64;
} tuning;

// Array initialization: mirrors the original C init_array
void init_array(auto path) {
	using namespace noarr;

	noarr::traverser(path).for_each([=](auto state) {
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
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(std::size_t n, auto path) {
	using namespace noarr;

	// A dummy 1D structure used only to traverse k in [0, n)
	auto k_struct = noarr::scalar<char>() ^ k_vec ^ noarr::set_length<'k'>(n);

#pragma scop
	// Outer loop: k = 0 .. n-1
	noarr::traverser(k_struct).for_each([&](auto sk) {
		const std::size_t k = get_index<'k'>(sk);

		// Views for the k-th row and k-th column:
		//   row_k[j] == path[k][j]
		//   col_k[i] == path[i][k]
		//
		// These are lightweight Noarr views; they do not copy data.
		auto row_k = path ^ noarr::fix<'i'>(k);
		auto col_k = path ^ noarr::fix<'j'>(k);

		// Loop over rows 'i'. For each row we cache path[i][k] once in 'pik'
		// and reuse it across all columns 'j'.
		noarr::traverser(path).template for_dims<'i'>([&](auto trav_i) {
			// trav_i has 'i' fixed and will iterate over all 'j'
			auto si = trav_i.state();          // contains current 'i' index
			const num_t pik = col_k[si];       // path[i][k], reused for the whole row

			// Inner loop over 'j' in cache-friendly tiles.
			// into_blocks_dynamic<'j', 'J', 'j', 'p'> splits the 'j'
			// dimension into:
			//   - 'J' : block index
			//   - 'j' : index within block
			//   - 'p' : presence flag for elements in the (possibly)
			//           incomplete last block
			//
			// Using it with traverser::order only changes traversal order;
			// the state seen inside the lambda still exposes the original
			// indices 'i' and 'j', so the update below is exactly the same
			// Floyd–Warshall recurrence.
			auto trav_blocked =
				trav_i.order(noarr::into_blocks_dynamic<'j', 'J', 'j', 'p'>(tuning.j_block_size));

			trav_blocked.for_each([&](auto sij) {
				// sij contains 'i' and 'j' indices for the original matrix
				num_t &pij = path[sij];          // path[i][j]
				const num_t via_k = pik + row_k[sij]; // path[i][k] + path[k][j]

				// Standard Floyd–Warshall relaxation:
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