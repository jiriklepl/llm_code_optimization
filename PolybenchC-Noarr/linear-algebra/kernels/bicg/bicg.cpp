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
[[gnu::flatten, gnu::noinline]]
void kernel_bicg(auto A, auto s, auto q, auto p, auto r) {
	using namespace noarr;

	#pragma scop
	// s[j] = 0 for all j
	traverser(s).for_each([=](auto state) {
		s[state] = SCALAR_VAL(0.0);
	});

	// For each i:
	//   q[i] = 0;
	//   for each j:
	//     s[j] += r[i] * A[i][j];
	//     q[i] += A[i][j] * p[j];
	traverser(A, s, q, p, r).template for_dims<'i'>([=](auto inner) {
		// state with current i fixed
		auto istate = inner.state();

		// q[i] = 0;
		q[istate] = SCALAR_VAL(0.0);

		// loop over j
		inner.for_each([=](auto state) {
			// state has 'i' and 'j'
			// s[j] += r[i] * A[i][j];
			s[state] += r[state] * A[state];

			// q[i] += A[i][j] * p[j];
			q[state] += A[state] * p[state];
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