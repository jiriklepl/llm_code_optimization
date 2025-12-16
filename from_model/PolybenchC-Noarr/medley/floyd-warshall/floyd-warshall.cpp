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
//   - keeps k as the outer, sequential DP dimension
//   - uses 1D scratch buffers col[i] and row[j] to snapshot column k and row k
//     at the beginning of each k-stage; this removes any potential intra‑stage
//     read‑after‑write hazards on path[*,k] and path[k,*] and improves locality
//   - tiles the (i,j) iteration space using into_blocks_dynamic + hoist to get
//     cache‑friendly 2D tiles while still iterating over logical (i,j) pairs
//     via the original 'i' and 'j' indices
//
[[gnu::flatten, gnu::noinline]]
void kernel_floyd_warshall(std::size_t n, auto path) {
	using namespace noarr;

	// Tiling along rows (i) and columns (j). These sizes are chosen for
	// cache and SIMD friendliness; they do not need to divide n, edge tiles
	// are handled automatically by into_blocks_dynamic.
	constexpr std::size_t tile_i = 32;
	constexpr std::size_t tile_j = 32;

	// Scratch buffers to snapshot the k-th column and row of 'path'.
	// Both are 1D, O(n) each, so the extra memory is negligible compared to
	// the n^2 cells of 'path'.
	auto col_buf = noarr::bag(noarr::scalar<num_t>() ^ i_vec ^ noarr::set_length<'i'>(n));
	auto row_buf = noarr::bag(noarr::scalar<num_t>() ^ j_vec ^ noarr::set_length<'j'>(n));

	auto col = col_buf.get_ref(); // col[i] will hold path[i, k]
	auto row = row_buf.get_ref(); // row[j] will hold path[k, j]

	// 1D helper structure used only to iterate k in [0, n).
	auto k_struct = noarr::scalar<char>() ^ k_vec ^ noarr::set_length<'k'>(n);

	// Proto-structure encoding a 2D tiling of the (i,j) plane:
	//  - split 'i' into tiles of height tile_i (I: tile index, i: intra-tile)
	//  - split 'j' into tiles of width  tile_j (J: tile index, j: intra-tile)
	//  - hoist I and J so that we iterate tiles first, then elements inside
	//
	// The into_blocks_dynamic variants handle edge tiles whose size is smaller
	// than tile_i / tile_j; guard dimensions ('r', 's') are internal when used
	// via traverser(...).order(...), so the states we see still only carry 'i'
	// and 'j'.
	auto tiled_ij =
		noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(tile_i) ^ noarr::hoist<'I'>() ^
		noarr::into_blocks_dynamic<'j', 'J', 'j', 's'>(tile_j) ^ noarr::hoist<'J'>();

#pragma scop
	// Outer DP loop over k remains strictly sequential.
	traverser(k_struct).for_each([=](auto sk) {
		auto k = get_index<'k'>(sk);

		// 1) Snapshot column k: col[i] = path[i, k] for all i.
		//    This represents d^{(k-1)}[i, k] in the standard FW recurrence.
		traverser(col).for_each([=](auto si) {
			auto i = get_index<'i'>(si);
			col[si] = path[idx<'i', 'j'>(i, k)];
		});

		// 2) Snapshot row k: row[j] = path[k, j] for all j.
		//    This represents d^{(k-1)}[k, j].
		traverser(row).for_each([=](auto sj) {
			auto j = get_index<'j'>(sj);
			row[sj] = path[idx<'i', 'j'>(k, j)];
		});

		// 3) Tiled relaxation:
		//       path[i, j] = min(path[i, j], col[i] + row[j])  for all (i, j).
		//
		// For this fixed k, col[] and row[] are read-only snapshots of the
		// previous DP level, so every (i,j) update is independent and may be
		// freely reordered. We traverse in a cache-friendly tile order
		// described by 'tiled_ij', but the state we see still carries only
		// logical indices 'i' and 'j'.
		traverser(path).order(tiled_ij).for_each([=](auto sij) {
			auto [i, j] = get_indices<'i', 'j'>(sij);

			const num_t via_k = col[idx<'i'>(i)] + row[idx<'j'>(j)];
			num_t &pij = path[sij];

			// Branchless min; equivalent to:
			//   path[i, j] = std::min(path[i, j], via_k);
			if (via_k < pij)
				pij = via_k;
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