#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "bicg.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// A: i x j (row-major: j is the inner/fast dimension)
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);

	// 1D vectors
	DEFINE_PROTO_STRUCT(s_layout, j_vec); // length M, index 'j'
	DEFINE_PROTO_STRUCT(q_layout, i_vec); // length N, index 'i'
	DEFINE_PROTO_STRUCT(p_layout, j_vec); // length M, index 'j'
	DEFINE_PROTO_STRUCT(r_layout, i_vec); // length N, index 'i'
} tuning;

// Array initialization
// A: i x j
// r: i
// p: j
void init_array(auto A, auto r, auto p) {
	using namespace noarr;

	// p[j] = (j % m) / m;
	traverser(p).for_each([=](auto state) {
		auto j = get_index<'j'>(state);
		auto m = p | get_length<'j'>();
		p[state] = (num_t)(j % m) / (num_t)m;
	});

	// r[i] = (i % n) / n;
	traverser(r).for_each([=](auto state) {
		auto i = get_index<'i'>(state);
		auto n = r | get_length<'i'>();
		r[state] = (num_t)(i % n) / (num_t)n;
	});

	// A[i][j] = (i * (j + 1) % n) / n;
	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		auto n = A | get_length<'i'>();
		A[state] = (num_t)(i * (j + 1) % n) / (num_t)n;
	});
}

// Main computational kernel
// A: i x j
// s: j
// q: i
// p: j
// r: i
//
// Semantics:
//   s[j] = sum_i r[i] * A[i,j]      (A^T * r)
//   q[i] = sum_j A[i,j] * p[j]      (A * p)
//
// Optimizations:
//   - Keep A row-major, traverse with j as the innermost dimension.
//   - 2D tiling of (i,j) using into_blocks_dynamic for better cache locality.
//   - Hoist r[i] outside the inner j-loop.
//   - Accumulate q[i] in a scalar register and write it once per row.
[[gnu::flatten, gnu::noinline]]
void kernel_bicg(auto A, auto s, auto q, auto p, auto r) {
	using namespace noarr;

	// Tunable tile sizes (compile-time constants here, can be adjusted for the target)
	// We use dynamic blocking to correctly handle sizes that are not multiples
	// of the tile sizes.
	constexpr auto tile_i = noarr::lit<64>;   // rows per tile
	constexpr auto tile_j = noarr::lit<128>;  // cols per tile

	// Split dimension 'i' into tiles 'I' (tile index) and 'i' (intra-tile),
	// with guard dimension 'R' indicating whether a given (I,i) is valid.
	// Likewise for 'j' into 'J' and 'j' with guard 'S'.
	auto blocked =
		noarr::into_blocks_dynamic<'i', 'I', 'i', 'R'>(tile_i) ^
		noarr::into_blocks_dynamic<'j', 'J', 'j', 'S'>(tile_j);

	#pragma scop
	// Phase 0: initialize s[j] = 0 for all j
	traverser(s).for_each([&](auto state) {
		s[state] = SCALAR_VAL(0.0);
	});

	// Phase 1: main BiCG kernel, blocked for cache locality
	//
	// Conceptual structure of the loops produced below:
	//   for each I tile (rows):
	//     for each valid row i in that tile:
	//       ri = r[i];
	//       qi = 0;
	//       for each J tile (columns):
	//         for each valid column j in that tile:
	//           a = A[i,j];
	//           s[j] += ri * a;
	//           qi   += a * p[j];
	//       q[i] = qi;
	//
	traverser(A, s, q, p, r)
		.order(blocked)
		// Outer loop over row tiles I
		.template for_dims<'I'>([&](auto trav_I) {
			// Loop over rows 'i' inside the tile, guarded by 'R' to skip
			// any out-of-bounds rows in the last (partial) tile.
			trav_I.template for_dims<'i', 'R'>([&](auto trav_iR) {
				// State that identifies the current row i (plus tile index I and guard R).
				// Extra dimensions beyond 'i' are ignored by 1D structures like r and q.
				auto row_state = trav_iR.state();

				// Hoist r[i] out of the inner j-loop: used for all columns of this row.
				num_t ri = r[row_state];

				// Register accumulator for q[i], to avoid repeated loads/stores.
				num_t qi = SCALAR_VAL(0.0);

				// Loop over column tiles J for this row.
				trav_iR.template for_dims<'J'>([&](auto trav_J) {
					// Loop over columns j inside the tile, guarded by 'S' so
				 // that the last (partial) tile is handled correctly.
					trav_J.template for_dims<'j', 'S'>([&](auto trav_ij) {
						// trav_ij has 'i' and 'j' (plus tile and guard dims).
						// All vector/matrix bags ignore unrelated dimensions.
						num_t a_ij = A[trav_ij];

						// s[j] += r[i] * A[i,j];
						s[trav_ij] += ri * a_ij;

						// q[i] += A[i,j] * p[j];
						qi += a_ij * p[trav_ij];
					});
				});

				// Finalize q[i] for this row i.
				q[row_state] = qi;
			});
		});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// Retrieve problem size
	std::size_t n = N; // length in 'i'
	std::size_t m = M; // length in 'j'

	// Common length proto-structure
	auto set_lengths =
		noarr::set_length<'i'>(n) ^
		noarr::set_length<'j'>(m);

	// Allocate bags with chosen layouts
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto s = noarr::bag(noarr::scalar<num_t>() ^ tuning.s_layout ^ set_lengths);
	auto q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ set_lengths);
	auto p = noarr::bag(noarr::scalar<num_t>() ^ tuning.p_layout ^ set_lengths);
	auto r = noarr::bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ set_lengths);

	// Initialize arrays
	init_array(A.get_ref(), r.get_ref(), p.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// Run kernel
	kernel_bicg(A.get_ref(), s.get_ref(), q.get_ref(), p.get_ref(), r.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// Print live-out data to prevent dead-code elimination
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, s.get_ref());
		noarr::serialize_data(std::cerr, q.get_ref());
	}

	// Print timing
	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;

	return 0;
}