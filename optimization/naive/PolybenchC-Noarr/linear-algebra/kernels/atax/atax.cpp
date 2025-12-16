#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include <noarr/traversers.hpp>
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

// Array initialization
void init_array(auto A, auto x) {
	using namespace noarr;

	// x: j dimension of length N
	const std::size_t n_len = x | get_length<'j'>();
	const num_t fn = static_cast<num_t>(n_len);
	const num_t inv_fn = static_cast<num_t>(1) / fn;

	// x[j] = 1 + (j / fn)
	traverser(x).for_each([&](auto state) {
		const std::size_t j = get_index<'j'>(state);
		x[state] = static_cast<num_t>(1) + static_cast<num_t>(j) * inv_fn;
	});

	// A: dimensions i (M) and j (N)
	const std::size_t m_len   = A | get_length<'i'>();
	const std::size_t n_len_A = A | get_length<'j'>();

	// Precompute reciprocal of (5 * m) to avoid a division in the innermost loop
	const num_t inv_5m = static_cast<num_t>(1) / (static_cast<num_t>(5) * static_cast<num_t>(m_len));

	// A[i][j] = ((i + j) % n) / (5 * m)
	traverser(A).for_each([&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		const num_t numerator = static_cast<num_t>((i + j) % n_len_A);
		A[state] = numerator * inv_5m;
	});
}

// Main computational kernel
//
// Original mathematical operation:
//   tmp[i] = sum_j A[i][j] * x[j]
//   y[j]   = sum_i A[i][j] * tmp[i] = (A^T * (A * x))[j]
//
// The implementation below keeps exactly this semantics, but is organized
// in three clearly separated phases to expose more optimization opportunities:
//   1) y = 0
//   2) tmp = A * x  (one matrix-vector product)
//   3) y  = A^T * tmp  (second matrix-vector product)
//
// Within each phase we:
//   - traverse rows 'i' with for_dims<'i'> to respect row-major locality of A
//   - accumulate per-row results in a scalar register (tmp_i) instead of
//     repeatedly reading/writing tmp[i] inside the inner loop
//   - use plain traverser.for_each(...) (no templated for_each<...>) in line
//     with the documented Noarr API
[[gnu::flatten, gnu::noinline]]
void kernel_atax(auto A, auto x, auto y, auto tmp) {
	using namespace noarr;

	#pragma scop

	// ---------------------------------------------------------------------
	// 1) y[j] = 0 for all j
	// ---------------------------------------------------------------------
	traverser(y).for_each([&](auto sy) {
		y[sy] = static_cast<num_t>(0);
	});

	// ---------------------------------------------------------------------
	// 2) tmp[i] = sum_j A[i][j] * x[j]
	//    Traverse rows 'i' and keep tmp[i] in a register (tmp_i) while
	//    accumulating over 'j'. This avoids repeated loads/stores of tmp[i]
	//    in the inner loop and makes the inner loop a clean reduction that
	//    is easy to auto-vectorize.
	// ---------------------------------------------------------------------
	traverser(A, x).template for_dims<'i'>([&](auto trav_i) {
		// State with fixed i (row index)
		auto si = trav_i.state();

		num_t tmp_i = static_cast<num_t>(0);

		// Iterate remaining dimension(s) – here it is just 'j'
		trav_i.for_each([&](auto s_ij) {
			// s_ij contains both 'i' (from si) and 'j'
			tmp_i += A[s_ij] * x[s_ij]; // x only uses 'j', extra dims are ignored
		});

		// Store the final accumulated value back to tmp[i]
		tmp[si] = tmp_i;
	});

	// ---------------------------------------------------------------------
	// 3) y[j] += A[i][j] * tmp[i]  for all i, j
	//    This is equivalent to y = A^T * tmp. We again traverse rows 'i'
	//    to preserve A's row-major access pattern, but we:
	//      - read tmp[i] once per row into a register (tmp_i)
	//      - use tmp_i inside the inner 'j' loop
	//    This reduces traffic to tmp and keeps the inner loop simple and
	//    friendly to vectorization and the cache.
	// ---------------------------------------------------------------------
	traverser(A, tmp, y).template for_dims<'i'>([&](auto trav_i) {
		// State with fixed i
		auto si = trav_i.state();

		// Load tmp[i] once per row
		const num_t tmp_i = tmp[si];

		// Inner loop over j
		trav_i.for_each([&](auto s_ij) {
			// s_ij again has 'i' and 'j'; y ignores 'i' and uses only 'j'
			y[s_ij] += A[s_ij] * tmp_i;
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