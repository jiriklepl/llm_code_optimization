/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Initialize A as in the original code. */
  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize B to zero.
   *
   * We replace the explicit double loop with memset on the
   * underlying contiguous data. This preserves semantics because
   * the representation of floating-point zero is all-zero bits.
   */
  memset((void*)POLYBENCH_ARRAY(B), 0,
         (size_t)n * (size_t)n * sizeof(DATA_TYPE));

  /* Compute B = A * A^T.
   *
   * Original code:
   *   for (t = 0; t < n; ++t)
   *     for (r = 0; r < n; ++r)
   *       for (s = 0; s < n; ++s)
   *         B[r][s] += A[r][t] * A[s][t];
   *
   * We keep the loop order (t, r, s) identical to preserve the
   * exact sequence of floating-point operations, but we:
   *   - Cache A[r][t] in a scalar (Art).
   *   - Cache the row pointer B[r] once per (t, r).
   * This reduces address arithmetic and memory traffic.
   */
  for (t = 0; t < n; ++t)
  {
    for (r = 0; r < n; ++r)
    {
      DATA_TYPE Art = A[r][t];
      DATA_TYPE * restrict Br = POLYBENCH_ARRAY(B)[r];
      for (s = 0; s < n; ++s)
      {
        Br[s] += Art * A[s][t];
      }
    }
  }

  /* Copy B back into A (same semantics as original). */
  for (r = 0; r < n; ++r)
  {
    DATA_TYPE * restrict Ar = A[r];
    DATA_TYPE * restrict Br = POLYBENCH_ARRAY(B)[r];
    for (s = 0; s < n; ++s)
      Ar[s] = Br[s];
  }

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
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky[[gnu::flatten, gnu::noinline]](int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  /* n is provided for consistency with the PolyBench interface,
     but the loop bound macro _PB_N is used as in the original code. */
  (void)n;

#pragma scop
  /* Optimized Cholesky factorization.
   *
   * Original structure:
   *   for (i)
   *     for (j < i)
   *       A[i][j] -= sum_{k<j} A[i][k] * A[j][k];
   *       A[i][j] /= A[j][j];
   *     A[i][i] -= sum_{k<i} A[i][k] * A[i][k];
   *     A[i][i] = sqrt(A[i][i]);
   *
   * Key optimizations:
   *   1. Use row pointers (Ai, Aj) to avoid repeated 2D indexing.
   *   2. Accumulate the dot product for A[i][j] in a scalar (sum)
   *      instead of updating A[i][j] in memory every iteration.
   *      The operations are performed in exactly the same order.
   *   3. Fuse the diagonal update with the j-loop using an
   *      accumulator 'diag'. This removes the separate k-loop for
   *      the diagonal while preserving the order of the
   *      diagonal-update operations.
   *   4. Manually unroll the inner k-loop by a factor of 4 to
   *      increase ILP and help vectorization. The sequence of
   *      floating-point operations remains the same
   *      (k = 0,1,2,3,4,...) as in the original loop.
   */
  for (int i = 0; i < _PB_N; ++i)
  {
    /* Pointer to row i of A. Marked restrict because no other
       pointer in this scope aliases this row. */
    DATA_TYPE * restrict Ai = A[i];

    /* Start from the original diagonal element A[i][i].
       This will be updated as:
         diag = A[i][i]_orig - sum_{j<i} (L[i][j]^2)
       matching the original algorithm's second loop. */
    DATA_TYPE diag = Ai[i];

    /* Off-diagonal elements in row i (j < i). */
    for (int j = 0; j < i; ++j)
    {
      DATA_TYPE * restrict Aj = A[j];

      /* Start from the original A[i][j] value. */
      DATA_TYPE sum = Ai[j];

      /* Compute sum -= Σ_{k=0}^{j-1} Ai[k] * Aj[k].
         Loop unrolled by 4; operation order is preserved. */
      int k = 0;
      for (; k + 3 < j; k += 4)
      {
        sum -= Ai[k]     * Aj[k];
        sum -= Ai[k + 1] * Aj[k + 1];
        sum -= Ai[k + 2] * Aj[k + 2];
        sum -= Ai[k + 3] * Aj[k + 3];
      }
      for (; k < j; ++k)
      {
        sum -= Ai[k] * Aj[k];
      }

      /* Divide by the diagonal element A[j][j] to get L[i][j]. */
      sum /= Aj[j];
      Ai[j] = sum;

      /* Update diagonal accumulator using the newly computed L[i][j].
         This matches the original diagonal loop:
           A[i][i] -= A[i][k] * A[i][k];   for k = 0..i-1
         but fused into the j-loop while preserving the order
         of these subtractions. */
      diag -= sum * sum;
    }

    /* Finalize diagonal element L[i][i]. */
    Ai[i] = SQRT_FUN(diag);
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
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

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