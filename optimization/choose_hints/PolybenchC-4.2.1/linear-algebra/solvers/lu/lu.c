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

  /* Alias the 2D array parameter with an explicit VLA pointer marked
   * restrict. This helps the compiler with alias analysis and can
   * enable better vectorization.
   */
  DATA_TYPE (*restrict a)[n] = A;

  /* Original lower-triangular initialization. */
  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	a[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	a[i][j] = 0;
      }
      a[i][i] = 1;
    }

  /* Make the matrix positive semi-definite.
   * (Same structure as the original code, but the Gram matrix
   *  A * A^T is computed more efficiently by exploiting symmetry.)
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  DATA_TYPE (*restrict b)[n] = POLYBENCH_ARRAY(B);

  /* Initialize B to zero. */
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      b[r][s] = (DATA_TYPE)0;

  /* Compute B = A * A^T.
   *
   * B[r][s] = sum_t A[r][t] * A[s][t]
   *
   * The result is symmetric: B[r][s] == B[s][r].
   * We therefore accumulate only the upper triangle (s >= r) and
   * mirror it afterwards. For each fixed (r,s), the sequence of
   * arithmetic operations on B[r][s] is the same as in the original
   * code (t from 0 to n-1 in order), so the numerical results for
   * each entry are preserved exactly.
   */
  for (t = 0; t < n; ++t)
  {
    for (r = 0; r < n; ++r)
    {
      const DATA_TYPE art = a[r][t];
      /* Update upper triangle B[r][s] for s >= r. */
      for (s = r; s < n; ++s)
      {
        b[r][s] += art * a[s][t];
      }
    }
  }

  /* Fill the lower triangle using symmetry: B[s][r] = B[r][s].
   * In the original algorithm B[r][s] and B[s][r] are mathematically
   * identical and are built from the same sequence of products
   * (just written twice). Copying preserves the final values while
   * cutting the work roughly in half.
   */
  for (r = 0; r < n; ++r)
    for (s = r+1; s < n; ++s)
      b[s][r] = b[r][s];

  /* Copy B back into A. */
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

  /* Alias the matrix with a restrict-qualified VLA pointer to improve
   * alias analysis and enable better vectorization and register usage.
   */
  DATA_TYPE (*restrict a)[n] = A;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *restrict Ai = a[i];

    /* Compute the L factors (below the diagonal) in row i.
     *
     * Original code:
     *   for (j = 0; j < i; j++) {
     *     for (k = 0; k < j; k++)
     *       A[i][j] -= A[i][k] * A[k][j];
     *     A[i][j] /= A[j][j];
     *   }
     *
     * We keep the same arithmetic order per element but use a local
     * accumulator 'sum' to avoid repeatedly loading/storing Ai[j].
     */
    for (j = 0; j < i; j++) {
      DATA_TYPE sum = Ai[j];
      for (k = 0; k < j; k++) {
        sum -= Ai[k] * a[k][j];
      }
      Ai[j] = sum / a[j][j];
    }

    /* Compute/update the U row (diagonal and above) in row i.
     *
     * Original code:
     *   for (j = i; j < _PB_N; j++) {
     *     for (k = 0; k < i; k++)
     *       A[i][j] -= A[i][k] * A[k][j];
     *   }
     *
     * This is mathematically:
     *   A[i][j] <- A[i][j] - sum_{k=0..i-1} A[i][k] * A[k][j]
     *
     * We rewrite it in an outer-product form:
     *
     *   for k in [0, i):
     *     for j in [i, N):
     *       A[i][j] -= A[i][k] * A[k][j];
     *
     * For each fixed j, the sequence of updates on A[i][j] is still
     * k = 0, 1, ..., i-1 in order, so the floating-point results
     * for each matrix entry are preserved. The difference is that we
     * now traverse A[k][i..N-1] contiguously for each k and reuse
     * A[i][k] across the whole inner loop, which greatly improves
     * cache locality and enables efficient SIMD vectorization.
     */
    for (k = 0; k < i; k++) {
      const DATA_TYPE lik = Ai[k];
      const DATA_TYPE *restrict Ak = a[k];

      /* The inner j-loop has no loop-carried dependencies: each Ai[j]
       * is updated exactly once per (i,k) pair and does not interact
       * with other j indices.  This pragma tells GCC it is safe to
       * vectorize even if it cannot prove independence on its own.
       */
#pragma GCC ivdep
      for (j = i; j < _PB_N; j++) {
        Ai[j] -= lik * Ak[j];
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