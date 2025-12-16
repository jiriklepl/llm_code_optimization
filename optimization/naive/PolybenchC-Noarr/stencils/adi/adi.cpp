#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
// Traverser iterator support (range-based for over traversers)
#include <noarr/structures/interop/traverser_iter.hpp>

// include common definitions
#include "defines.hpp"

// include benchmark-specific definitions
#include "adi.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	// First apply j_vec (inner), then i_vec (outer) to match C layout u[i][j]
	DEFINE_PROTO_STRUCT(u_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(v_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(p_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(q_layout, j_vec ^ i_vec);
} tuning;


// initialization function
void init_array(auto U) {
	using namespace noarr;

	auto n = U | get_length<'i'>();

	// U: i x j
	traverser(U).for_each([=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		U[state] = (num_t)(i + n - j) / (num_t)n;
	});
}


// computation kernel
//
// Optimizations applied:
//
// 1. Precompute reusable slice proto-structures for the interior of the domain
//    so we do not rebuild them repeatedly inside the inner loops.
//
// 2. Use traverser iterators (range-based for) for the outer 'i' loops. This
//    keeps the iteration logic in the Noarr world but gives us a clear outer
//    loop body where we could attach OpenMP pragmas if desired.
//
// 3. Reuse the same inner traverser (over 'j') for both the forward and
//    backward sweeps, avoiding repeated construction of identical traversal
//    configurations.
//
// 4. Keep all multidimensional iteration expressed via Noarr traversers and
//    states. The only explicit numeric loop is over the time dimension 't'.
[[gnu::flatten, gnu::noinline]]
void kernel_adi(std::size_t tsteps, auto U, auto V, auto P, auto Q) {
	using namespace noarr;

	auto n = U | get_length<'i'>();

	num_t DX = 0, DY = 0, DT = 0;
	num_t B1 = 0, B2 = 0;
	num_t mul1 = 0, mul2 = 0;
	num_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;

#pragma scop

	DX = SCALAR_VAL(1.0) / (num_t)n;
	DY = SCALAR_VAL(1.0) / (num_t)n;
	DT = SCALAR_VAL(1.0) / (num_t)tsteps;
	B1 = SCALAR_VAL(2.0);
	B2 = SCALAR_VAL(1.0);
	mul1 = B1 * DT / (DX * DX);
	mul2 = B2 * DT / (DY * DY);

	a = -mul1 / SCALAR_VAL(2.0);
	b = SCALAR_VAL(1.0) + mul1;
	c = a;
	d = -mul2 / SCALAR_VAL(2.0);
	e = SCALAR_VAL(1.0) + mul2;
	f = d;

	// Reusable interior slices (1 .. n-2) for both dimensions.
	// These are proto-structures; applying them via .order(...)
	// does not change the physical layout, only the traversed subset.
	auto interior_i = noarr::slice<'i'>(1, n - 2);
	auto interior_j = noarr::slice<'j'>(1, n - 2);

	for (std::size_t t = 1; t <= tsteps; t++) {

		// -------------------------------------------------------------------
		// Column sweep
		//
		// We keep the outer loop over 'i' and the inner sweep over 'j',
		// exactly as in the original code, but expressed using:
		//   - a range-based for over a traverser (outer 'i'),
		//   - .for_each over an inner traverser (inner 'j').
		// -------------------------------------------------------------------
		{
			// Traverser over all four bags, restricted to interior i in [1, n-2].
			auto col_trav = traverser(U, V, P, Q).order(interior_i);

			// Each iteration of this loop corresponds to one fixed i.
			// This form is also suitable for OpenMP:
			//   #pragma omp parallel for
			//   for (auto trav_i : col_trav) { ... }
			for (auto trav_i : col_trav) {
				auto state_i = trav_i.state();
				auto i = get_index<'i'>(state_i);

				// Boundary conditions for column i
				V[noarr::idx<'i', 'j'>(0,     i)] = SCALAR_VAL(1.0);
				P[noarr::idx<'i', 'j'>(i,     0)] = SCALAR_VAL(0.0);
				Q[noarr::idx<'i', 'j'>(i,     0)] = V[noarr::idx<'i', 'j'>(0, i)];

				// Inner traverser restricted to interior j in [1, n-2] for this i.
				auto col_inner_trav = trav_i.order(interior_j);

				// Forward sweep in j: j = 1 .. n-2
				col_inner_trav.for_each([&U, &V, &P, &Q, a, b, c, d, f, n, i](auto state) {
					// state carries both 'i' and 'j' for this column, with i fixed
					auto prev = state - noarr::idx<'j'>(1);
					auto j = get_index<'j'>(state);

					const num_t denom = a * P[prev] + b;
					P[state] = -c / denom;

					// Note: U is accessed with indices (j, i-1), (j, i), (j, i+1),
					// exactly as in the original code. We keep this pattern to
					// preserve semantics.
					const num_t u_j_im1 = U[noarr::idx<'i', 'j'>(j, i - 1)];
					const num_t u_j_i   = U[noarr::idx<'i', 'j'>(j, i    )];
					const num_t u_j_ip1 = U[noarr::idx<'i', 'j'>(j, i + 1)];

					Q[state] =
						(-d * u_j_im1
						 + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u_j_i
						 - f * u_j_ip1
						 - a * Q[prev]) / denom;
				});

				// Boundary at bottom row for this column
				V[noarr::idx<'i', 'j'>(n - 1, i)] = SCALAR_VAL(1.0);

				// Backward sweep in j: j = n-2 .. 1
				//
				// Implemented as a forward sweep over k = 1 .. n-2 with
				//    j = n - 1 - k,
				// so that we still iterate the underlying index sequence in the
				// natural order produced by the traverser, but apply the
				// recurrence in the required reverse-j order.
				col_inner_trav.for_each([&V, &P, &Q, n, i](auto state) {
					auto k = get_index<'j'>(state);
					std::size_t j = n - 1 - k; // n-2, n-3, ..., 1

					auto pij_state   = noarr::idx<'i', 'j'>(i,     j);
					auto vjp1i_state = noarr::idx<'i', 'j'>(j + 1, i);

					V[noarr::idx<'i', 'j'>(j, i)] =
						P[pij_state] * V[vjp1i_state] + Q[pij_state];
				});
			}
		}

		// -------------------------------------------------------------------
		// Row sweep
		//
		// Symmetric to the column sweep, but now the tridiagonal systems
		// are along rows. Again, outer loop over 'i', inner sweep over 'j'.
		// -------------------------------------------------------------------
		{
			// Traverser over all four bags, restricted to interior i in [1, n-2].
			auto row_trav = traverser(U, V, P, Q).order(interior_i);

			for (auto trav_i : row_trav) {
				auto state_i = trav_i.state();
				auto i = get_index<'i'>(state_i);

				// Boundary conditions for row i
				U[noarr::idx<'i', 'j'>(i, 0)] = SCALAR_VAL(1.0);
				P[noarr::idx<'i', 'j'>(i, 0)] = SCALAR_VAL(0.0);
				Q[noarr::idx<'i', 'j'>(i, 0)] = U[noarr::idx<'i', 'j'>(i, 0)];

				// Inner traverser restricted to interior j in [1, n-2] for this i.
				auto row_inner_trav = trav_i.order(interior_j);

				// Forward sweep in j: j = 1 .. n-2
				row_inner_trav.for_each([&U, &V, &P, &Q, a, c, d, e, f, n, i](auto state) {
					auto prev = state - noarr::idx<'j'>(1);
					auto j = get_index<'j'>(state);

					const num_t denom = d * P[prev] + e;
					P[state] = -f / denom;

					const num_t v_im1_j = V[noarr::idx<'i', 'j'>(i - 1, j)];
					const num_t v_i_j   = V[noarr::idx<'i', 'j'>(i,     j)];
					const num_t v_ip1_j = V[noarr::idx<'i', 'j'>(i + 1, j)];

					Q[state] =
						(-a * v_im1_j
						 + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v_i_j
						 - c * v_ip1_j
						 - d * Q[prev]) / denom;
				});

				// Boundary at rightmost column for this row
				U[noarr::idx<'i', 'j'>(i, n - 1)] = SCALAR_VAL(1.0);

				// Backward sweep in j: j = n-2 .. 1
				row_inner_trav.for_each([&U, &P, &Q, n, i](auto state) {
					auto k = get_index<'j'>(state);
					std::size_t j = n - 1 - k; // n-2, ..., 1

					auto pij_state   = noarr::idx<'i', 'j'>(i, j);
					auto uijp1_state = noarr::idx<'i', 'j'>(i, j + 1);

					U[pij_state] = P[pij_state] * U[uijp1_state] + Q[pij_state];
				});
			}
		}
	}

#pragma endscop
}

} // namespace


int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t tsteps = TSTEPS;

	// input data
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n);

	auto U = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ set_lengths);
	auto V = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ set_lengths);
	auto P = noarr::bag(noarr::scalar<num_t>() ^ tuning.p_layout ^ set_lengths);
	auto Q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ set_lengths);

	// initialize data
	init_array(U.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_adi(tsteps, U.get_ref(), V.get_ref(), P.get_ref(), Q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cerr << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cerr, U.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << duration.count() << std::endl;
}