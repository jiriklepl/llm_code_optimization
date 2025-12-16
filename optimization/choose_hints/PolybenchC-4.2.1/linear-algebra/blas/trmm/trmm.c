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


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;

  /* Precompute reciprocals once instead of doing a division in each
     iteration. This preserves the original semantics while reducing
     the cost of the initialization. */
  const DATA_TYPE inv_m = (DATA_TYPE)1.0 / (DATA_TYPE)m;
  const DATA_TYPE inv_n = (DATA_TYPE)1.0 / (DATA_TYPE)n;

  for (i = 0; i < m; i++) {
    /* Initialize strictly lower triangular part of A. */
    for (j = 0; j < i; j++) {
      DATA_TYPE v = (DATA_TYPE)((i + j) % m);
      A[i][j] = v * inv_m;
    }

    /* Unit diagonal. */
    A[i][i] = (DATA_TYPE)1.0;

    /* Initialize B row-wise (contiguous in memory). */
    for (j = 0; j < n; j++) {
      DATA_TYPE v = (DATA_TYPE)((n + (i - j)) % n);
      B[i][j] = v * inv_n;
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
static
void __attribute__((flatten, noinline))
kernel_trmm(int m, int n,
	     DATA_TYPE alpha,
	     DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
	     DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /* m and n are unused inside this kernel body because PolyBench
     provides the (possibly padded) problem sizes via _PB_M/_PB_N. */
  (void)m;
  (void)n;

  /* Create local restrict-qualified aliases. This tells the compiler
     that A and B do not alias, which helps vectorization and
     hoisting of loads/stores. */
  DATA_TYPE (*restrict A_)[M] = A;
  DATA_TYPE (*restrict B_)[N] = B;

  const int m_pb = _PB_M;
  const int n_pb = _PB_N;

//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM (lower unit-triangular), B is MxN
#pragma scop
  /* Optimized loop structure:
   *
   * Original code:
   *   for i
   *     for j
   *       for k = i+1..M-1
   *         B[i][j] += A[k][i] * B[k][j];
   *       B[i][j] = alpha * B[i][j];
   *
   * We transform this to:
   *   for i
   *     for k = i+1..M-1
   *       load aik = A[k][i] once
   *       for j
   *         B[i][j] += aik * B[k][j];
   *     for j
   *       B[i][j] *= alpha;
   *
   * This preserves the exact per-element operation order over k,
   * but makes the innermost loop run over j with unit stride
   * on both B[i][:] and B[k][:]. This greatly improves cache
   * locality and enables efficient SIMD vectorization.
   */
  for (i = 0; i < m_pb; i++)
  {
    /* Pointer to the current output row to avoid repeated address
       arithmetic in the inner loops. */
    DATA_TYPE *restrict Bi = B_[i];

    /* Accumulate contributions from rows k > i into row i. */
    for (k = i + 1; k < m_pb; k++)
    {
      const DATA_TYPE aik = A_[k][i];           /* reused across all j */
      const DATA_TYPE *restrict Bk = B_[k];     /* row k of B */

      /* Inner loop has unit-stride accesses in both Bi and Bk, which
         is ideal for auto-vectorization and cache usage. */
      for (j = 0; j < n_pb; j++)
      {
        Bi[j] += aik * Bk[j];
      }
    }

    /* Scale entire row i by alpha.
       Kept separate to retain the original semantics:
       B[i][j] = alpha * ( original B[i][j] + sum_k A[k][i]*B[k][j] ). */
    if (alpha != (DATA_TYPE)1.0)
    {
      for (j = 0; j < n_pb; j++)
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