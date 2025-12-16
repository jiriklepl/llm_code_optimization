#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "adi.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(u_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(v_layout, i_vec ^ j_vec);
	DEFINE_PROTO_STRUCT(p_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(q_layout, j_vec ^ i_vec);
} tuning;

// initialization function
void init_array(auto u) {
	// u: i x j
	using namespace noarr;

	auto n = u | get_length<'i'>();

	traverser(u) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);

		u[state] = (num_t)(i + n - j) / n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_adi(auto tsteps, auto u, auto v, auto p, auto q) {
	// u: i x j
	// v: j x i
	// p: i x j
	// q: i x j
	using namespace noarr;

	auto u_trans = u ^ rename<'i', 'j', 'j', 'i'>();
	auto v_trans = v ^ rename<'i', 'j', 'j', 'i'>();
	auto trav = traverser(u, v, p, q) ^ bcast<'t'>(tsteps);

	#pragma scop

	num_t DX = (num_t)1.0 / (u | get_length<'i'>());
	num_t DY = (num_t)1.0 / (u | get_length<'j'>());
	num_t DT = (num_t)1.0 / tsteps;

	num_t B1 = 2.0;
	num_t B2 = 1.0;

	num_t mul1 = B1 * DT / (DX * DX);
	num_t mul2 = B2 * DT / (DY * DY);

	num_t a = -mul1 / (num_t)2.0;
	num_t b = (num_t)1.0 + mul1;
	num_t c = a;

	num_t d = -mul2 / (num_t)2.0;
	num_t e = (num_t)1.0 + mul2;
	num_t f = d;

	trav ^ symmetric_spans<'i', 'j'>(u, 1, 1) | for_dims<'t'>([=](auto inner) {
		// column sweep
		inner | for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			v[state & idx<'j'>(0)] = (num_t)1.0;
			p[state & idx<'j'>(0)] = (num_t)0.0;
			q[state & idx<'j'>(0)] = v[state & idx<'j'>(0)];

			inner | [=](auto state) {
				p[state] = -c / (a * p[state - idx<'j'>(1)] + b);
				q[state] = (-d * u_trans[state - idx<'i'>(1)] + (B2 + B1 * d) * u_trans[state] -
								f * u_trans[state + idx<'i'>(1)] -
								a * q[state - idx<'j'>(1)]) /
							(a * p[state - idx<'j'>(1)] + b);
			};

			v[state & idx<'j'>((v | get_length<'j'>()) - 1)] = (num_t)1.0;

			inner ^ reverse<'j'>() | [=](auto state) {
				v[state] = p[state] * v[state + idx<'j'>(1)] + q[state];
			};
		});

		// row sweep
		inner | for_dims<'i'>([=](auto inner) {
			auto state = inner.state();

			u[state & idx<'j'>(0)] = (num_t)1.0;
			p[state & idx<'j'>(0)] = (num_t)0.0;
			q[state & idx<'j'>(0)] = u[state & idx<'j'>(0)];

			inner | [=](auto state) {
				p[state] = -f / (d * p[state - idx<'j'>(1)] + e);
				q[state] = (-a * v_trans[state - idx<'i'>(1)] + (B2 + B1 * a) * v_trans[state] -
								c * v_trans[state + idx<'i'>(1)] -
								d * q[state - idx<'j'>(1)]) /
							(d * p[state - idx<'j'>(1)] + e);
			};

			u[state & idx<'j'>((u | get_length<'j'>()) - 1)] = (num_t)1.0;

			inner ^ reverse<'j'>() | [=](auto state) {
				u[state] = p[state] * u[state + idx<'j'>(1)] + q[state];
			};
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N;
	std::size_t t = TSTEPS;

	// data
	auto u = noarr::bag(noarr::scalar<num_t>() ^ tuning.u_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto v = noarr::bag(noarr::scalar<num_t>() ^ tuning.v_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto p = noarr::bag(noarr::scalar<num_t>() ^ tuning.p_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));
	auto q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(n));

	// initialize data
	init_array(u.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_adi(t, u.get_ref(), v.get_ref(), p.get_ref(), q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, u.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
