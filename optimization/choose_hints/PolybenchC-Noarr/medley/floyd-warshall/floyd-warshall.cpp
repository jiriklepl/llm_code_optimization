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
// Optimized version:
//
// - Still keeps k as the outermost loop to preserve the dynamic-programming
//   semantics of Floyd–Warshall.
// - Uses Noarr traversers for all three logical loops (k, i, j).
// - Exploits the row-major layout (j is the inner dimension) to traverse
//   rows contiguously and work with raw pointers inside the inner loops.
// - For a fixed k and i:
//     * path[i][k] is read once and reused for all j.
//     * The entire k-th row path[k][*] is reused.
//   This reduces address recalculation and memory traffic substantially.
//
// Correctness note:
// init_array() produces strictly positive weights (1..7) or 999, so all
// path[u][v] remain non‑negative during the computation. In particular:
//   - path[k][k] >= 0 for all k.
//   - For a fixed k, updating path[i][k] inside the j–loop computes
//       path[i][k] = min(path[i][k], path[i][k] + path[k][k])
//     which never decreases path[i][k]. Therefore, path[i][k] and the
//     whole row path[k][*] are invariant during the j-loop for a fixed k,
//     and it is safe to cache them once per (k, i).
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(std::size_t n, auto path) {
	using namespace noarr;

	// A dummy 1D structure used only to traverse k in [0, n)
	auto k_struct = noarr::scalar<char>() ^ k_vec ^ noarr::set_length<'k'>(n);

	// Raw pointer to the underlying data. Given tuning.path_layout = j_vec ^ i_vec
	// and the lengths set to (n, n), the layout is:
	//   path[i][j]  ==  data[i * n + j]
	auto *data = static_cast<num_t *>(path.data());

#pragma scop
	// Outer loop: k = 0 .. n-1
	traverser(k_struct).for_each([=](auto sk) {
		std::size_t k = get_index<'k'>(sk);

		// Pointer to the k-th row: path[k][0 .. n-1]
		num_t *row_k = data + k * n;

		// Middle loop: iterate rows i. We fix 'i' with for_dims<'i'> and then
		// traverse remaining dimension(s) with for_each (only 'j' remains).
		traverser(path).template for_dims<'i'>([=](auto t_i) {
			// t_i has 'i' fixed; its state knows which row we are on.
			auto s_i = t_i.state();
			std::size_t i = get_index<'i'>(s_i);

			// Pointer to the i-th row: path[i][0 .. n-1]
			num_t *row_i = data + i * n;

			// path[i][k] is constant for this (k, i) pair
			const num_t dik = row_i[k];

			// Inner loop over j for this fixed (k, i).
			// We advance two pointers in lockstep:
			//   pij -> path[i][j]
			//   pkj -> path[k][j]
			num_t *pij = row_i;
			num_t *pkj = row_k;

			t_i.for_each([&](auto /*sij*/) {
				const num_t via_k = dik + *pkj;
				num_t &cur = *pij;

				if (via_k < cur)
					cur = via_k;

				// Move to the next column j (row‑major contiguous access)
				++pij;
				++pkj;
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