/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"

/* Tunable blocking factor along the column (N) dimension.
 *
 * This controls how many columns of B are processed together in the
 * innermost loops of the kernel.  It can be adjusted at compile time
 * to better match the cache hierarchy:
 *
 *   - smaller values  -> smaller working set per block, potentially
 *                        better L1 data cache behavior for very wide B
 *   - larger values   -> fewer loop iterations/branches
 *
 * Override with e.g.:
 *   -DTRMM_J_BLOCK_SIZE=128
 *
 * Must be a positive integer.
 */
#ifndef TRMM_J_BLOCK_SIZE
# define TRMM_J_BLOCK_SIZE 64
#endif


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */

/* Use GCC attributes to keep the kernel body visible for optimization
 * while preventing inlining into main (PolyBench timing requirement). */
static void __attribute__((flatten, noinline))
kernel_trmm(int m, int n,
	    DATA_TYPE alpha,
	    DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
	    DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k, jj;

  /* BLAS parameters
   * SIDE   = 'L'
   * UPLO   = 'L'
   * TRANSA = 'T'
   * DIAG   = 'U'
   * => Form  B := alpha*A**T*B.
   * A is MxM (lower triangular, unit diagonal)
   * B is MxN
   *
   * Original reference kernel (for one (i,j)):
   *
   *   for (k = i+1; k < M; ++k)
   *     B[i][j] += A[k][i] * B[k][j];
   *   B[i][j] = alpha * B[i][j];
   *
   * Optimizations applied below:
   *   - Reorder loops to (i, k, j) with column blocking (jj, j) so that
   *     the innermost loop walks contiguous memory along the row of B.
   *     This turns the inner computation into a cache-friendly AXPY
   *     pattern:
   *         B[i][:] += A[k][i] * B[k][:]
   *   - Introduce a tunable blocking factor TRMM_J_BLOCK_SIZE on the
   *     column dimension to keep the working set per block in cache.
   *   - Use temporary row pointers (Bi, Bk) marked restrict to help
   *     the compiler with alias analysis and vectorization.
   *   - Use #pragma GCC ivdep on the innermost loops: there are no
   *     loop-carried dependencies in j, so this is safe and encourages
   *     SIMD code generation.
   *
   * The per-element arithmetic is unchanged:
   *   B[i][j]_final = alpha * ( B[i][j]_orig
   *                             + sum_{k=i+1..M-1} A[k][i] * B[k][j]_orig )
   * and the order of summation over k for each (i,j) remains increasing
   * in k, as in the original code.
   */

#pragma scop
  for (i = 0; i < _PB_M; i++)
  {
    /* Pointer to row i of B; restrict is valid because within this
     * iteration we never access row i through any other pointer. */
    DATA_TYPE *restrict Bi = B[i];

    /* Process row i in column blocks to improve cache locality
     * when N is large. */
    for (jj = 0; jj < _PB_N; jj += TRMM_J_BLOCK_SIZE)
    {
      int j_end = jj + TRMM_J_BLOCK_SIZE;
      if (j_end > _PB_N)
        j_end = _PB_N;

      /* Triangular multiplication:
       *   for all k > i:
       *     Bi[j] += A[k][i] * Bk[j],  j in [jj, j_end)
       *
       * Loop order (i, k, j) gives contiguous access to both Bi and Bk
       * across j, which is significantly more cache- and SIMD-friendly
       * than the original (i, j, k) ordering that walked B by columns.
       */
      for (k = i + 1; k < _PB_M; k++)
      {
        const DATA_TYPE aik = A[k][i];
        const DATA_TYPE *restrict Bk = B[k]; /* row k of B, read-only here */

        /* No loop-carried dependencies in j: each Bi[j] is independent.
         * This pragma allows GCC to safely vectorize the loop even under
         * strict aliasing rules. */
        #pragma GCC ivdep
        for (j = jj; j < j_end; j++)
        {
          Bi[j] += aik * Bk[j];
        }
      }

      /* Scale the updated block of row i by alpha.
       * For each (i,j) this multiply happens after all contributions
       * from k > i have been accumulated, exactly as in the original
       * kernel; only the loop nesting has changed. */
      #pragma GCC ivdep
      for (j = jj; j < j_end; j++)
      {
        Bi[j] *= alpha;
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm (m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}