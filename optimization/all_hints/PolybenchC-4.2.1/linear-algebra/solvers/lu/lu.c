/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Initial lower-triangular matrix with 1s on the diagonal. */
  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite.
   * (Same construction as in the cholesky benchmark.)
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Use local restrict-qualified aliases for better optimization. */
  DATA_TYPE (* restrict a)[n] = A;
  DATA_TYPE (* restrict b)[n] = POLYBENCH_ARRAY(B);

  /* Zero-initialize B. */
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      b[r][s] = (DATA_TYPE)0;

  /* Compute B = A * A^T in the original loop order (t, r, s),
   * but hoist invariant loads to improve locality and reduce
   * redundant memory accesses. Arithmetic order is unchanged.
   */
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      {
        DATA_TYPE art = a[r][t];          /* reused across inner s-loop */
        DATA_TYPE * restrict b_row = b[r];

        for (s = 0; s < n; ++s)
          b_row[s] += art * a[s][t];
      }

  /* Copy back to A. */
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      a[r][s] = b[r][s];

  POLYBENCH_FREE_ARRAY(B);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_lu[[gnu::flatten, gnu::noinline]](int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

  /* Local restrict-qualified alias of A to help the compiler
   * reason about aliasing and optimize memory accesses.
   */
  DATA_TYPE (* restrict a)[n] = A;

  /* Tunable blocking factor for the update of the upper-triangular
   * part (columns). Chosen to fit well in typical L1/L2 caches on
   * recent x86-64 CPUs; can be adjusted if needed.
   */
  const int J_BLOCK = 64;

  /* Minimum number of columns remaining on a row before we use
   * OpenMP parallelization for the U update. This avoids the
   * overhead of parallel regions for very small workloads.
   */
  const int MIN_PARALLEL_COLS = 128;

#pragma scop
  for (i = 0; i < _PB_N; i++) {

    DATA_TYPE * restrict a_i = a[i];

    /* --------------------------------------------------------------
     * Step 1: Compute L factors on row i (columns 0 .. i-1).
     *
     * Original code:
     *   for (j = 0; j < i; j++) {
     *     for (k = 0; k < j; k++)
     *       A[i][j] -= A[i][k] * A[k][j];
     *     A[i][j] /= A[j][j];
     *   }
     *
     * We keep exactly the same arithmetic order but use a temporary
     * accumulator "sum" in a register to reduce memory traffic.
     * -------------------------------------------------------------- */
    for (j = 0; j < i; j++) {
      DATA_TYPE sum = a_i[j];

      for (k = 0; k < j; k++) {
        sum -= a_i[k] * a[k][j];
      }

      a_i[j] = sum / a[j][j];
    }

    /* --------------------------------------------------------------
     * Step 2: Update U factors on row i (columns i .. N-1).
     *
     * Original code:
     *   for (j = i; j < N; j++) {
     *     for (k = 0; k < i; k++) {
     *       A[i][j] -= A[i][k] * A[k][j];
     *     }
     *   }
     *
     * Observations:
     *   - For fixed i, all columns j >= i are independent.
     *   - For each such column j, the update is:
     *       A[i][j] -= sum_{k=0}^{i-1} A[i][k] * A[k][j];
     *
     * Optimization strategy:
     *   - Block the column dimension j to improve cache locality.
     *   - Within each block we iterate k outer / j inner so that
     *     both a_i[j] and a[k][j] are accessed with unit stride.
     *   - Parallelize over j-blocks with OpenMP: each block works
     *     on a disjoint subset of columns, so there are no races.
     *
     * The mathematical result is the same as the original code
     * (modulo floating-point round-off differences due to reordering
     * of independent operations).
     * -------------------------------------------------------------- */

    int cols = _PB_N - i;

    if (cols >= MIN_PARALLEL_COLS) {
      /* Parallel path: distribute column blocks across threads. */
      #pragma omp parallel for schedule(static)
      for (int jj = i; jj < _PB_N; jj += J_BLOCK) {
        int j_end = jj + J_BLOCK;
        if (j_end > _PB_N)
          j_end = _PB_N;

        for (int k_local = 0; k_local < i; ++k_local) {
          DATA_TYPE aik = a_i[k_local];
          DATA_TYPE * restrict a_k = a[k_local];

          /* Inner j-loop is free of loop-carried dependencies,
           * so it can be safely vectorized.
           */
          #pragma omp simd
          for (int j_local = jj; j_local < j_end; ++j_local) {
            a_i[j_local] -= aik * a_k[j_local];
          }
        }
      }
    } else {
      /* Serial path: same computation but without OpenMP overhead. */
      for (int jj = i; jj < _PB_N; jj += J_BLOCK) {
        int j_end = jj + J_BLOCK;
        if (j_end > _PB_N)
          j_end = _PB_N;

        for (int k_local = 0; k_local < i; ++k_local) {
          DATA_TYPE aik = a_i[k_local];
          DATA_TYPE * restrict a_k = a[k_local];

          #pragma omp simd
          for (int j_local = jj; j_local < j_end; ++j_local) {
            a_i[j_local] -= aik * a_k[j_local];
          }
        }
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_lu (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}