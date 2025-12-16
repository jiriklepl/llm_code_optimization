#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "defines.hpp"
#include "bicg.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();

struct tuning {
	DEFINE_PROTO_STRUCT(a_layout, j_vec ^ i_vec); // A: i x j (row-major: j inner)
	DEFINE_PROTO_STRUCT(s_layout, j_vec);         // s: j
	DEFINE_PROTO_STRUCT(q_layout, i_vec);         // q: i
	DEFINE_PROTO_STRUCT(p_layout, j_vec);         // p: j
	DEFINE_PROTO_STRUCT(r_layout, i_vec);         // r: i
} tuning;

// initialization function
void init_array(int /*m*/, int /*n*/, auto A, auto r, auto p) {
	using namespace noarr;

	// p[j] = (j % m)/m
	traverser(p) | [=](auto sj) {
		auto j = get_index<'j'>(sj);
		auto m = p | get_length<'j'>();
		p[sj] = (num_t)(j % m) / (num_t)m;
	};

	// r[i] = (i % n)/n
	traverser(r) | [=](auto si) {
		auto i = get_index<'i'>(si);
		auto n = r | get_length<'i'>();
		r[si] = (num_t)(i % n) / (num_t)n;
	};

	// A[i][j] = (i*(j+1) % n)/n
	traverser(A) | [=](auto sij) {
		auto [i, j] = get_indices<'i', 'j'>(sij);
		auto n = A | get_length<'i'>();
		A[sij] = (num_t)((i * (j + 1)) % n) / (num_t)n;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_bicg(auto A, auto s, auto q, auto p, auto r) {
	using namespace noarr;

	#pragma scop
	// s[j] = 0
	traverser(s) | [=](auto sj) {
		s[sj] = (num_t)0;
	};

	// for (i) { q[i]=0; for (j) { s[j]+=r[i]*A[i][j]; q[i]+=A[i][j]*p[j]; } }
	traverser(A, p, q, r, s) | for_dims<'i'>([=](auto inner) {
		auto si = inner.state(); // contains fixed 'i'

		q[si] = (num_t)0;

		inner | for_each<'j'>([=](auto sij) {
			s[sij] += r[si] * A[sij];
			q[si] += A[sij] * p[sij];
		});
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t n = N; // length in 'i'
	std::size_t m = M; // length in 'j'

	// set lengths proto
	auto set_lengths = noarr::set_length<'i'>(n) ^ noarr::set_length<'j'>(m);

	// bags
	auto A = noarr::bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto s = noarr::bag(noarr::scalar<num_t>() ^ tuning.s_layout ^ set_lengths);
	auto q = noarr::bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ set_lengths);
	auto p = noarr::bag(noarr::scalar<num_t>() ^ tuning.p_layout ^ set_lengths);
	auto r = noarr::bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ set_lengths);

	// initialize data
	init_array((int)m, (int)n, A.get_ref(), r.get_ref(), p.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_bicg(A.get_ref(), s.get_ref(), q.get_ref(), p.get_ref(), r.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, s.get_ref());
		noarr::serialize_data(std::cout, q.get_ref());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
