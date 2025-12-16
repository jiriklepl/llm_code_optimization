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
static
void kernel_trmm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /* Use local copies of the (possibly macro-based) problem sizes to
     avoid re-evaluating macros and to give the compiler clearer
     loop bounds. */
  const int Mdim = _PB_M;
  const int Ndim = _PB_N;

//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
#pragma scop
  /* Optimized loop nest.

     Original computation:

       for (i = 0; i < _PB_M; i++)
         for (j = 0; j < _PB_N; j++) {
           for (k = i+1; k < _PB_M; k++)
             B[i][j] += A[k][i] * B[k][j];
           B[i][j] = alpha * B[i][j];
         }

     Transformations applied:

       1. Loop interchange of the j and k loops
          ------------------------------------------------
          For each (i, j), the sum over k is unchanged and
          still executed in strictly increasing k order.
          We instead iterate over k first and j inside, so
          for a fixed pair (i, k) we traverse complete rows
          B[i][*] and B[k][*] contiguously:

             Bi[j] += A[k][i] * Bk[j],  j = 0..Ndim-1

          This improves cache locality and enables efficient
          SIMD vectorization, since j is now the innermost
          loop and corresponds to the contiguous dimension
          of B in memory.

       2. Address-calculation reduction
          ------------------------------------------------
          We create row pointers Bi and Bk to avoid repeated
          2D index computations B[i][j] and B[k][j] inside
          the innermost loop.

       3. Scaling kept per-row, after accumulation
          ------------------------------------------------
          For each row i, we first accumulate all
          contributions from rows k > i into B[i][*], then
          scale that row by alpha, exactly as in the
          original code. This preserves the arithmetic
          order for each element (sum over k, then one
          multiplication by alpha).

       The combination of these transformations preserves
       the exact mathematical result (including the order
       of the reductions over k for each (i, j)) while
       significantly improving data locality and making the
       innermost loops easier to vectorize.
  */

  for (i = 0; i < Mdim; ++i)
  {
    /* Pointer to the current row of B that we update. */
    DATA_TYPE *Bi = B[i];

    /* Triangular matrix multiply part: Bi += sum_{k>i} A[k][i] * Bk. */
    for (k = i + 1; k < Mdim; ++k)
    {
      const DATA_TYPE aik = A[k][i]; /* Single load reused across all columns. */
      DATA_TYPE *Bk = B[k];

      /* Inner-most loop over columns (contiguous dimension of B). */
      #pragma GCC ivdep
      for (j = 0; j < Ndim; ++j)
      {
        Bi[j] += aik * Bk[j];
      }
    }

    /* Scale row i of B by alpha, as in the original kernel. */
    if (alpha != (DATA_TYPE)1)
    {
      #pragma GCC ivdep
      for (j = 0; j < Ndim; ++j)
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