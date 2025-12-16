#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "adi.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto x_vec = noarr::vector<'x'>(); // first index in C code (rows)
constexpr auto y_vec = noarr::vector<'y'>(); // second index in C code (cols)

struct tuning {
	DEFINE_PROTO_STRUCT(layout, y_vec ^ x_vec); // C layout: a[i][j]
} tuning;

// initialization function
void init_array(std::size_t n, auto U) {
	using namespace noarr;

	traverser(U) | [=](auto s) {
		auto [x, y] = get_indices<'x', 'y'>(s);
		U[s] = (num_t)(x + n - y) / (num_t)n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_adi(int tsteps, std::size_t n, auto U, auto V, auto P, auto Q) {
	using namespace noarr;

	#pragma scop
	num_t DX = (num_t)1.0 / (num_t)n;
	num_t DY = (num_t)1.0 / (num_t)n;
	num_t DT = (num_t)1.0 / (num_t)tsteps;
	num_t B1 = (num_t)2.0;
	num_t B2 = (num_t)1.0;
	num_t mul1 = B1 * DT / (DX * DX);
	num_t mul2 = B2 * DT / (DY * DY);

	num_t a = -mul1 / (num_t)2.0;
	num_t b = (num_t)1.0 + mul1;
	num_t c = a;
	num_t d = -mul2 / (num_t)2.0;
	num_t e = (num_t)1.0 + mul2;
	num_t f = d;

	for(int t = 1; t <= tsteps; t++) {
		// Column Sweep: iterate columns y = 1..n-2
		traverser(U, V, P, Q)
		.order(noarr::slice<'y'>((std::size_t)1, n - 2))
		| for_dims<'y'>([&](auto by_col) {
			auto yc = noarr::get_index<'y'>(by_col.state());

			// boundaries
			V[noarr::idx<'x', 'y'>(0, yc)] = (num_t)1.0;
			P[noarr::idx<'x', 'y'>(yc, 0)] = (num_t)0.0;
			Q[noarr::idx<'x', 'y'>(yc, 0)] = V[noarr::idx<'x', 'y'>(0, yc)];

			// forward sweep over rows x = 1..n-2
			by_col.order(noarr::slice<'x'>((std::size_t)1, n - 2)).for_each([&](auto s) {
				auto xr = noarr::get_index<'x'>(s);

				num_t denom = a * P[noarr::idx<'x', 'y'>(yc, xr - 1)] + b;
				P[noarr::idx<'x', 'y'>(yc, xr)] = -c / denom;

				num_t rhs =
					(-d * U[noarr::idx<'x', 'y'>(xr, yc - 1)]
					+ ((num_t)1.0 + (num_t)2.0 * d) * U[noarr::idx<'x', 'y'>(xr, yc)]
					- f * U[noarr::idx<'x', 'y'>(xr, yc + 1)]
					- a * Q[noarr::idx<'x', 'y'>(yc, xr - 1)]);
				Q[noarr::idx<'x', 'y'>(yc, xr)] = rhs / denom;
			});

			// boundary
			V[noarr::idx<'x', 'y'>(n - 1, yc)] = (num_t)1.0;

			// backward sweep over rows x = n-2..1
			std::size_t cnt = 0;
			by_col.order(noarr::slice<'x'>((std::size_t)1, n - 2)).for_each([&](auto) {
				std::size_t xr = (n - 2) - cnt; // descending index
				V[noarr::idx<'x', 'y'>(xr, yc)] =
					P[noarr::idx<'x', 'y'>(yc, xr)] * V[noarr::idx<'x', 'y'>(xr + 1, yc)]
					+ Q[noarr::idx<'x', 'y'>(yc, xr)];
				cnt++;
			});
		});

		// Row Sweep: iterate rows x = 1..n-2
		traverser(U, V, P, Q)
		.order(noarr::slice<'x'>((std::size_t)1, n - 2))
		| for_dims<'x'>([&](auto by_row) {
			auto xr = noarr::get_index<'x'>(by_row.state());

			// boundaries
			U[noarr::idx<'x', 'y'>(xr, 0)] = (num_t)1.0;
			P[noarr::idx<'x', 'y'>(xr, 0)] = (num_t)0.0;
			Q[noarr::idx<'x', 'y'>(xr, 0)] = U[noarr::idx<'x', 'y'>(xr, 0)];

			// forward sweep over cols y = 1..n-2
			by_row.order(noarr::slice<'y'>((std::size_t)1, n - 2)).for_each([&](auto s) {
				auto yc = noarr::get_index<'y'>(s);

				num_t denom = d * P[noarr::idx<'x', 'y'>(xr, yc - 1)] + e;
				P[noarr::idx<'x', 'y'>(xr, yc)] = -f / denom;

				num_t rhs =
					(-a * V[noarr::idx<'x', 'y'>(xr - 1, yc)]
					+ ((num_t)1.0 + (num_t)2.0 * a) * V[noarr::idx<'x', 'y'>(xr, yc)]
					- c * V[noarr::idx<'x', 'y'>(xr + 1, yc)]
					- d * Q[noarr::idx<'x', 'y'>(xr, yc - 1)]);
				Q[noarr::idx<'x', 'y'>(xr, yc)] = rhs / denom;
			});

			// boundary
			U[noarr::idx<'x', 'y'>(xr, n - 1)] = (num_t)1.0;

			// backward sweep over cols y = n-2..1
			std::size_t cnt = 0;
			by_row.order(noarr::slice<'y'>((std::size_t)1, n - 2)).for_each([&](auto) {
				std::size_t yc = (n - 2) - cnt; // descending index
				U[noarr::idx<'x', 'y'>(xr, yc)] =
					P[noarr::idx<'x', 'y'>(xr, yc)] * U[noarr::idx<'x', 'y'>(xr, yc + 1)]
					+ Q[noarr::idx<'x', 'y'>(xr, yc)];
				cnt++;
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
	int tsteps = TSTEPS;

	auto set_lengths = noarr::set_length<'x'>(n) ^ noarr::set_length<'y'>(n);

	auto U = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);
	auto V = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);
	auto P = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);
	auto Q = noarr::bag(noarr::scalar<num_t>() ^ tuning.layout ^ set_lengths);

	// initialize data
	init_array(n, U.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_adi(tsteps, n, U.get_ref(), V.get_ref(), P.get_ref(), Q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, U.get_ref() ^ noarr::hoist<'x'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
