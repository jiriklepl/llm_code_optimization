#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "mvt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// 1D layouts for vectors (all use dimension 'i')
	DEFINE_PROTO_STRUCT(x_layout, i_vec);

	// 2D layout for matrix A: dimensions 'i' (rows) and 'j' (columns),
	// with 'j' the inner/fastest-varying dimension (row-major).
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	auto n = x1 | get_length<'i'>();

	// Initialize the four vectors
	traverser(x1, x2, y_1, y_2).for_each([=](auto state) {
		auto i = get_index<'i'>(state);

		x1[state]  = (num_t)((i % n)) / n;
		x2[state]  = (num_t)(((i + 1) % n)) / n;
		y_1[state] = (num_t)(((i + 3) % n)) / n;
		y_2[state] = (num_t)(((i + 4) % n)) / n;
	});

	// Initialize matrix A: A[i][j] = (i * j % n) / n
	auto nA = A | get_length<'i'>();

	traverser(A).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (num_t)((i * j) % nA) / nA;
	});
}

// computation kernel
//
// Original mathematical behaviour:
//
//   // phase 1
//   for i in [0, N):
//     for j in [0, N):
//       x1[i] += A[i, j] * y_1[j];
//
//   // phase 2
//   for i in [0, N):
//     for j in [0, N):
//       x2[i] += A[j, i] * y_2[j];
//
// The implementation below keeps exactly these computations, but:
//
//   - Phase 1 keeps x1[i] in a register accumulator while reducing over j,
//     so x1[i] is loaded/stored only once per i.
//   - Phase 2 is re-ordered to iterate j outermost and i innermost while
//     using a transposed view A_t. This makes accesses to A_t[j, i]
//     contiguous in memory (good data locality) and loads y_2[j] only
//     once per outer j-iteration.
//
[[gnu::flatten, gnu::noinline]]
void kernel_mvt(auto x1, auto x2, auto y_1, auto y_2, auto A) {
	// x1, x2, y_1, y_2: 1D over 'i'
	// A: 2D over 'i' x 'j'
	using namespace noarr;

	// Views of y_1 and y_2 as vectors over 'j' (needed for y_[*][j])
	auto y_1_j = y_1.get_ref() ^ noarr::rename<'i', 'j'>();
	auto y_2_j = y_2.get_ref() ^ noarr::rename<'i', 'j'>();

	// Transposed logical view of A to access A[j][i] as A_t[i][j].
	// Layout in memory is unchanged; only the mapping of ('i','j') names
	// is swapped so that A_t[i,j] == A[j,i].
	auto A_t = A.get_ref() ^ noarr::rename<'i', 'j', 'j', 'i'>();

#pragma scop
	// ------------------------------------------------------------------
	// Phase 1:
	//   for i in [0, N):
	//     x1[i] += sum_j A[i, j] * y_1[j]
	//
	// We use:
	//   - an outer traverser over 'i',
	//   - a register accumulator per i,
	//   - an inner j-loop that walks the row-major (j-innermost) layout
	//     of A contiguously.
	// ------------------------------------------------------------------
	noarr::traverser(x1, y_1_j, A).template for_dims<'i'>([&](auto trav_i) {
		// State with current 'i' fixed, no 'j' yet.
		auto si = trav_i.state(); // contains only index_in<'i'>

		// Register accumulator for x1[i]: load once, update in inner loop,
		// store once.
		num_t acc = x1[si];

		// Iterate j as the innermost dimension. For A this means walking
		// A[i, j] with unit stride in memory (row-major).
		trav_i.template for_each<'j'>([&](auto s) {
			// s has indices 'i' and 'j'
			acc += A[s] * y_1_j[s];
		});

		// Write back the accumulated value for this i.
		x1[si] = acc;
	});

	// ------------------------------------------------------------------
	// Phase 2:
	//   for i in [0, N):
	//     x2[i] += sum_j A[j, i] * y_2[j]
	//
	// The straightforward implementation (i outer, j inner) with A[j, i]
	// would access A with a stride of N and is cache-unfriendly.
	//
	// Instead, we:
	//   - use the transposed view A_t such that A_t[i, j] = A[j, i],
	//   - iterate j outermost, i innermost:
	//         for j:
	//           y = y_2[j]
	//           for i:
	//             x2[i] += A_t[i, j] * y
	//     which walks each logical "row" of A_t (corresponding to a row of
	//     the original A) contiguously along its inner 'i' dimension.
	// ------------------------------------------------------------------
	noarr::traverser(x2, y_2_j, A_t).template for_dims<'j'>([&](auto trav_j) {
		// State with current 'j' fixed
		auto sj = trav_j.state(); // contains only index_in<'j'>

		// Load y_2[j] once per j-iteration and reuse in the inner loop.
		num_t y_val = y_2_j[sj];

		// Now iterate 'i' as the innermost dimension. For A_t this is the
		// contiguous (fast-varying) dimension, so A_t[s] is read with good
		// spatial locality. x2[i] is updated in place.
		trav_j.template for_each<'i'>([&](auto s) {
			// s has indices 'i' and 'j'
			x2[s] = x2[s] + A_t[s] * y_val;
		});
	});
#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;

	// common length proto-structure for all dimensions
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	// data structures
	auto A   = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto x1  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto x2  = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_1 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);
	auto y_2 = noarr::bag(noarr::scalar<num_t>() ^ tuning.x_layout ^ set_lengths);

	// initialize data
	init_array(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_mvt(x1.get_ref(), x2.get_ref(), y_1.get_ref(), y_2.get_ref(), A.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results (prevents dead-code elimination), if requested
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, x1.get_ref());
		noarr::serialize_data(std::cerr, x2.get_ref());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}