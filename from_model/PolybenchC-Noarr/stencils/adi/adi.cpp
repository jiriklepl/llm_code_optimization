#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

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
// Optimizations compared to the original version:
//
//   * Use 1D temporaries (P_col/Q_col, P_row/Q_row) for the Thomas
//     algorithm instead of the full 2D P/Q grids. This keeps all
//     tridiagonal coefficients in cache / registers and removes
//     O(N^2) redundant memory traffic to P/Q.
//     The P and Q bags are kept in the signature to preserve the
//     original interface, but are no longer used.
//
//   * Precompute combinations c1 = 1 + 2*d and c2 = 1 + 2*a and use
//     them in the inner loops to avoid recomputing these expressions
//     at every grid point.
//
//   * Strength reduction for divisions: compute inv_denom = 1/denom
//     once per iteration and use multiplications instead of two
//     separate divisions.
//
//   * All spatial loop nests remain expressed via Noarr traversers,
//     so the iteration order and data access structure are explicit
//     and optimizable by the compiler.
[[gnu::flatten, gnu::noinline]]
void kernel_adi(std::size_t tsteps, auto U, auto V, auto P, auto Q) {
	using namespace noarr;

	auto n = U | get_length<'i'>();

	// P and Q are now unused inside the kernel body; they are kept only
	// to preserve the original function interface and allocation pattern.
	(void)P;
	(void)Q;

#pragma scop

	// --------------------------------------------------------------------
	// Discretization / scheme coefficients (identical math as original)
	// --------------------------------------------------------------------
	const num_t DX = SCALAR_VAL(1.0) / (num_t)n;
	const num_t DY = SCALAR_VAL(1.0) / (num_t)n;
	const num_t DT = SCALAR_VAL(1.0) / (num_t)tsteps;
	const num_t B1 = SCALAR_VAL(2.0);
	const num_t B2 = SCALAR_VAL(1.0);

	const num_t mul1 = B1 * DT / (DX * DX);
	const num_t mul2 = B2 * DT / (DY * DY);

	const num_t a = -mul1 / SCALAR_VAL(2.0);
	const num_t b = SCALAR_VAL(1.0) + mul1;
	const num_t c = a;
	const num_t d = -mul2 / SCALAR_VAL(2.0);
	const num_t e = SCALAR_VAL(1.0) + mul2;
	const num_t f = d;

	// Loop-invariant combinations used inside the inner sweeps:
	// c1 = 1 + 2*d   (column sweep center coefficient)
	// c2 = 1 + 2*a   (row sweep center coefficient)
	const num_t c1 = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d;
	const num_t c2 = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a;

	// 1D temporaries for the Thomas algorithm along j.
	// Each has length n and is reused for every i-line.
	std::vector<num_t> P_col(n), Q_col(n);
	std::vector<num_t> P_row(n), Q_row(n);

	for (std::size_t t = 1; t <= tsteps; t++) {
		// ================================================================
		// Column sweep: implicit solve along j for each interior column i
		// ================================================================
		traverser(U, V)
			.order(slice<'i'>(1, n - 2)) // i = 1 .. n-2 (interior)
			.template for_dims<'i'>([&](auto trav_i) {
				auto i_state = trav_i.state();
				const std::size_t i = get_index<'i'>(i_state);

				// --- Top boundary for this column i: j = 0 ---
				V[idx<'i', 'j'>(0, i)] = SCALAR_VAL(1.0);
				P_col[0] = SCALAR_VAL(0.0);
				Q_col[0] = V[idx<'i', 'j'>(0, i)];

				// Forward Thomas sweep along j: j = 1 .. n-2
				auto trav_ij = trav_i.order(slice<'j'>(1, n - 2));
				trav_ij.for_each([&](auto state) {
					const std::size_t j = get_index<'j'>(state); // 1 .. n-2

					// Tridiagonal coefficients recurrence
					const num_t denom = a * P_col[j - 1] + b;
					const num_t inv_denom = SCALAR_VAL(1.0) / denom;
					P_col[j] = -c * inv_denom;

					// Stencil in the transverse (i) direction, matching the
					// original code:
					//   u_j_im1 = U[j, i-1];
					//   u_j_i   = U[j, i  ];
					//   u_j_ip1 = U[j, i+1];
					const num_t u_j_im1 = U[idx<'i', 'j'>(j, i - 1)];
					const num_t u_j_i   = U[idx<'i', 'j'>(j, i)];
					const num_t u_j_ip1 = U[idx<'i', 'j'>(j, i + 1)];

					const num_t rhs =
						(-d * u_j_im1) +
						(c1 * u_j_i)   -
						(f  * u_j_ip1) -
						(a  * Q_col[j - 1]);

					Q_col[j] = rhs * inv_denom;
				});

				// --- Bottom boundary for this column i: j = n-1 ---
				V[idx<'i', 'j'>(n - 1, i)] = SCALAR_VAL(1.0);

				// Backward substitution in j: logically j = n-2 .. 1.
				// We iterate the same j-states (1..n-2) but map them to
				// decreasing physical indices j_bwd = n-1-k, exactly as in
				// the original kernel.
				trav_ij.for_each([&](auto state) {
					const std::size_t k = get_index<'j'>(state);
					const std::size_t j_bwd = n - 1 - k; // n-2, n-3, ..., 1

					V[idx<'i', 'j'>(j_bwd, i)] =
						P_col[j_bwd] * V[idx<'i', 'j'>(j_bwd + 1, i)] +
						Q_col[j_bwd];
				});
			});

		// ================================================================
		// Row sweep: implicit solve along j for each interior row i
		// ================================================================
		traverser(U, V)
			.order(slice<'i'>(1, n - 2)) // i = 1 .. n-2 (interior)
			.template for_dims<'i'>([&](auto trav_i) {
				auto i_state = trav_i.state();
				const std::size_t i = get_index<'i'>(i_state);

				// --- Left boundary for this row i: j = 0 ---
				U[idx<'i', 'j'>(i, 0)] = SCALAR_VAL(1.0);
				P_row[0] = SCALAR_VAL(0.0);
				Q_row[0] = U[idx<'i', 'j'>(i, 0)];

				// Forward Thomas sweep along j: j = 1 .. n-2
				auto trav_ij = trav_i.order(slice<'j'>(1, n - 2));
				trav_ij.for_each([&](auto state) {
					const std::size_t j = get_index<'j'>(state); // 1 .. n-2

					const num_t denom = d * P_row[j - 1] + e;
					const num_t inv_denom = SCALAR_VAL(1.0) / denom;
					P_row[j] = -f * inv_denom;

					const num_t v_im1_j = V[idx<'i', 'j'>(i - 1, j)];
					const num_t v_i_j   = V[idx<'i', 'j'>(i,     j)];
					const num_t v_ip1_j = V[idx<'i', 'j'>(i + 1, j)];

					const num_t rhs =
						(-a * v_im1_j) +
						(c2 * v_i_j)   -
						(c  * v_ip1_j) -
						(d  * Q_row[j - 1]);

					Q_row[j] = rhs * inv_denom;
				});

				// --- Right boundary for this row i: j = n-1 ---
				U[idx<'i', 'j'>(i, n - 1)] = SCALAR_VAL(1.0);

				// Backward substitution in j: j = n-2 .. 1
				trav_ij.for_each([&](auto state) {
					const std::size_t k = get_index<'j'>(state);
					const std::size_t j_bwd = n - 1 - k; // n-2, ..., 1

					U[idx<'i', 'j'>(i, j_bwd)] =
						P_row[j_bwd] * U[idx<'i', 'j'>(i, j_bwd + 1)] +
						Q_row[j_bwd];
				});
			});
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