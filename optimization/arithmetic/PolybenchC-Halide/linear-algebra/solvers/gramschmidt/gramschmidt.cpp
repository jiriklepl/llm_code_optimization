#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.hpp"
#include "gramschmidt.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

struct tuning {
	constexpr static auto a_layout = j_vec ^ i_vec; // A: i x j (M x N)
	constexpr static auto q_layout = j_vec ^ i_vec; // Q: i x j (M x N)
	constexpr static auto r_layout = j_vec ^ k_vec; // R: k x j (N x N), first index is 'k'
} tuning;

// initialization function
void init_array(auto A, auto R, auto Q) {
	using namespace noarr;

	auto m = A | get_length<'i'>();
	auto n = A | get_length<'j'>();

	traverser(A, Q) | [&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		A[state] = (((num_t)((i * j) % m) / (num_t)m) * (num_t)100) + (num_t)10;
		Q[state] = (num_t)0;
	};

	traverser(R) | [&](auto state) {
		R[state] = (num_t)0;
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gramschmidt(auto A, auto R, auto Q) {
	using namespace noarr;

	auto n = A | get_length<'j'>();

	#pragma scop
	traverser(R) | for_dims<'k'>([&](auto trav_k) {
		auto k = get_index<'k'>(trav_k.state());

		// nrm = sum_i A[i][k]^2
		num_t nrm = (num_t)0;
		traverser(A).order(noarr::fix<'j'>(k)) | [&](auto si) {
			num_t a_ik = A[si];
			nrm += a_ik * a_ik;
		};

		// R[k][k] = sqrt(nrm)
		num_t rkk = std::sqrt(nrm);
		R[noarr::idx<'k', 'j'>(k, k)] = rkk;

		// Q[i][k] = A[i][k] / R[k][k]
		traverser(A, Q).order(noarr::fix<'j'>(k)) | [&](auto si) {
			Q[si] = A[si] / rkk;
		};

		// for j = k+1 .. n-1
		if(k + 1 < n) {
			auto tail_len = n - (k + 1);

			traverser(R).order(noarr::fix<'k'>(k) ^ noarr::slice<'j'>(k + 1, tail_len)) | [&](auto sj) {
				auto j = get_index<'j'>(sj);

				// R[k][j] = 0; R[k][j] += sum_i Q[i][k] * A[i][j]
				num_t rkj = (num_t)0;

				traverser(A, Q).order(noarr::fix<'j'>(j)) | [&](auto si) {
					auto i = get_index<'i'>(si);
					num_t q_ik = Q[noarr::idx<'i', 'j'>(i, k)];
					num_t a_ij = A[si];
					rkj += q_ik * a_ij;
				};

				R[sj] = rkj;

				// A[i][j] = A[i][j] - Q[i][k] * R[k][j]
				traverser(A, Q).order(noarr::fix<'j'>(j)) | [&](auto si) {
					auto i = get_index<'i'>(si);
					A[si] = A[si] - Q[noarr::idx<'i', 'j'>(i, k)] * rkj;
				};
			};
		}
	});
	#pragma endscop
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	// problem size
	std::size_t m = M;
	std::size_t n = N;

	// set lengths
	auto set_lengths = noarr::set_length<'i'>(m) ^ noarr::set_length<'j'>(n) ^ noarr::set_length<'k'>(n);

	// bags
	auto A = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths);
	auto R = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.r_layout ^ set_lengths);
	auto Q = noarr::make_bag(noarr::scalar<num_t>() ^ tuning.q_layout ^ set_lengths);

	// initialize data
	init_array(A.get_ref(), R.get_ref(), Q.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernel
	kernel_gramschmidt(A.get_ref(), R.get_ref(), Q.get_ref());

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<long double>(end - start);

	// print results (R then Q)
	if (argc > 0 && argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, R.get_ref() ^ noarr::hoist<'k'>());
		noarr::serialize_data(std::cout, Q.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << std::fixed << std::setprecision(6);
	std::cerr << duration.count() << std::endl;
}
