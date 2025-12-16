#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "atax.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A: m x n, row-major (i = row, j = column)
	DEFINE_PROTO_STRUCT(A_layout, j_vec ^ i_vec);
	// x: length n (dimension 'j')
	DEFINE_PROTO_STRUCT(x_layout, j_vec);
	// y: length n (dimension 'j')
	DEFINE_PROTO_STRUCT(y_layout, j_vec);
	// tmp: length m (dimension 'i')
	DEFINE_PROTO_STRUCT(tmp_layout, i_vec);
} tuning;

// -----------------------------------------------------------------------------
// Tiling configuration
// -----------------------------------------------------------------------------
// Tile size along the 'i' (row) dimension. This value is chosen so that a tile
// of rows fits comfortably in cache on a modern x64 CPU; it can be tuned.
constexpr std::size_t I_BLOCK = 64;

// Proto-structure that splits dimension 'i' into blocks:
//   - 'I'  : index of the row-block
//   - 'i'  : index within the block
//   - 'r'  : guard dimension indicating whether the row is present in this block
//
// We use the _dynamic_ variant so that we do not require M to be a multiple of
// I_BLOCK and still keep a regular tiled traversal. Rows beyond the real size
// simply get an empty 'r' and are not iterated.
constexpr auto i_blocking_proto =
	noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(noarr::lit<I_BLOCK>);

// -----------------------------------------------------------------------------
// Array initialization
// -----------------------------------------------------------------------------
void init_array(auto A, auto x) {
	using namespace noarr;

	// x: dimension 'j' has length N
	auto n_len = x | get_length<'j'>();
	num_t fn = static_cast<num_t>(n_len);
	// Precompute reciprocal once to avoid a division per element
	num_t inv_n = static_cast<num_t>(1) / fn;

	// x[j] = 1 + (j / fn)  ->  x[j] = 1 + j * inv_n
	traverser(x).for_each([=](auto state) {
		auto j = get_index<'j'>(state);
		x[state] = static_cast<num_t>(1) + static_cast<num_t>(j) * inv_n;
	});

	// A: dimensions 'i' (M) and 'j' (N)
	auto m_len = A | get_length<'i'>();
	auto n_len_A = A | get_length<'j'>(); // should equal n_len

	// A[i][j] = ((i + j) % n) / (5 * m)
	// Precompute reciprocal of (5 * m) to replace division by multiplication.
	num_t inv_5m = static_cast<num_t>(1) / static_cast<num_t>(5 * m_len);

	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		num_t numerator = static_cast<num_t>((i + j) % n_len_A);
		A[state] = numerator * inv_5m;
	});
}

// -----------------------------------------------------------------------------
// Main computational kernel
//
// Mathematical kernel:
//   tmp[i] = sum_j A[i,j] * x[j]
//   y[j]   = sum_i A[i,j] * tmp[i]
//
// Implementation strategy:
//   - Zero y[j] once.
//   - Traverse A row-by-row, but do so in tiles of rows ('I' blocks) to improve
//     cache/TLB behaviour on large M.
//   - For each concrete row i:
//       * compute tmp[i] using a scalar accumulator 'acc' (kept in a register);
//       * then, with the final acc = tmp[i], traverse the same row again to
//         update y[j].
//
// The original code already uses this fused per-row scheme; here we add
// tile-based traversal on 'i' to improve locality, and we keep the inner
// 'j' loops contiguous to help auto-vectorization.
// -----------------------------------------------------------------------------
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) {
	using namespace noarr;

	#pragma scop
	// -------------------------------------------------------------------------
	// y[j] = 0 for all j
	// -------------------------------------------------------------------------
	traverser(y).for_each([&](auto sy) {
		y[sy] = static_cast<num_t>(0);
	});

	// -------------------------------------------------------------------------
	// Tiled traversal over rows ('i' dimension)
	//
	// We apply i_blocking_proto only as a *view* via traverser.order(...),
	// leaving the physical layout untouched. The resulting dimensions are:
	//   - 'I' : tile index in i
	//   - 'r' : guard dimension (length 0 or 1)
	//   - 'i' : index within tile
	//   - 'j' : original column dimension (inner, contiguous)
	//
	// The nested for_dims below implement:
	//   for each tile I
	//     for each valid row i in this tile
	//       acc = 0
	//       // Phase 1: tmp[i] = sum_j A[i,j] * x[j]
	//       for all j:
	//         acc += A[i,j] * x[j]
	//       tmp[i] = acc
	//       // Phase 2: y[j] += A[i,j] * tmp[i]
	//       for all j:
	//         y[j] += A[i,j] * acc
	// -------------------------------------------------------------------------
	traverser(A, x, y, tmp)
		.order(i_blocking_proto)
		// Outer loop over row tiles
		.template for_dims<'I'>([&](auto trav_I) {
			// Loop over valid rows within this tile.
			// 'r' is the presence dimension created by into_blocks_dynamic:
			// if a given (I, i) is outside [0, M), its 'r' length is 0 and this
			// body is never executed for that combination.
			trav_I.template for_dims<'r', 'i'>([&](auto trav_row) {
				// State with fixed row index; contains ('I', 'r', 'i') and no 'j'.
				auto s_i = trav_row.state();

				// -----------------------------
				// Phase 1: tmp[i] = sum_j A[i,j] * x[j]
				// -----------------------------
				num_t acc = static_cast<num_t>(0);

				// After fixing (I, r, i), the only remaining dimension is 'j'.
				// This for_each walks j in unit stride, giving good memory
				// access patterns for A[i,j], x[j], and y[j].
				trav_row.for_each([&](auto s_ij) {
					acc += A[s_ij] * x[s_ij];
				});

				// Store the fully accumulated tmp[i]
				tmp[s_i] = acc;

				// -----------------------------
				// Phase 2: y[j] += A[i,j] * tmp[i]
				// -----------------------------
				trav_row.for_each([&](auto s_ij) {
					y[s_ij] += A[s_ij] * acc;
				});
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

	// common length proto-structure
	auto set_lengths = noarr::set_length<'i'>(m) ^ noarr::set_length<'j'>(n);

	// data structures
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.A_layout   ^ set_lengths);
	auto x   = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout   ^ set_lengths);
	auto y   = noarr::bag(noarr::scalar<num_t>() ^ tuning.y_layout   ^ set_lengths);
	auto tmp = noarr::bag(noarr::scalar<num_t>() ^ tuning.tmp_layout ^ set_lengths);

	// initialize arrays
	init_array(A.get_ref(), x.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_atax(A.get_ref(), x.get_ref(), y.get_ref(), tmp.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print result vector y
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, y.get_ref());
	}

	// print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}