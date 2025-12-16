/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimized version notes (semantics preserved):

   Original kernel structure:
     for i
       for j <= i:   C[i][j] *= beta;
       for k
         for j <= i:
           C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];

   Issues:
     - Poor locality for A[j][k] / B[j][k]: j varies fastest, but matrices
       are row-major => large-stride accesses along rows.
     - C[i][j] is loaded/stored once for scaling and then updated again
       in the k-loop.
     - Multiplication by 'alpha' happens twice per (i,j,k).

   This optimized version:
     - Fuses the scaling and update for each C[i][j] so that C[i][j] is
       kept in a scalar register across the entire k-reduction
       (one load and one store per element).
     - Reorders loops to make k the innermost dimension for A[*][k] and
       B[*][k], giving unit-stride, cache-friendly, vectorizable access.
     - For each row i, precomputes temporary rows:
           alphaA[k] = alpha * A[i][k]
           alphaB[k] = alpha * B[i][k]
       and reuses them across all j <= i.
       This hoists the expensive products with alpha out of the inner
       (j,k) loops and avoids reloading A[i][k] / B[i][k] for every j.
     - Only the lower triangle j <= i is touched, as in the original code.
*/
static
void kernel_syr2k[[gnu::flatten, gnu::noinline]](int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n) __restrict,
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m) __restrict,
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m) __restrict)
{
  int i, j, k;

  /* Local copies to help the compiler keep scalars in registers. */
  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

  /* Temporary buffers for alpha-scaled copies of row i of A and B.
     Size is O(M); they are reused for each row i. */
  DATA_TYPE alphaA[_PB_M];
  DATA_TYPE alphaB[_PB_M];

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  for (i = 0; i < _PB_N; i++)
  {
    /* Precompute alpha * A[i, k] and alpha * B[i, k] once per row i.
       Accesses A[i][k] and B[i][k] are contiguous in memory (row-major),
       which is friendly to caches and SIMD. */
#pragma GCC ivdep
    for (k = 0; k < _PB_M; k++)
    {
      DATA_TYPE aik = A[i][k];
      DATA_TYPE bik = B[i][k];
      alphaA[k] = alpha_local * aik;
      alphaB[k] = alpha_local * bik;
    }

    /* Update only the lower triangle (j <= i) of row i of C.

       For each (i, j) with j <= i, we compute:
         C[i, j] = beta * C[i, j] +
                   sum_k ( A[j, k] * alphaB[k] +
                           B[j, k] * alphaA[k] );

       This is algebraically equivalent to the original update:
         C[i, j] = beta * C[i, j] +
                   sum_k ( A[j, k]*alpha*B[i, k] +
                           B[j, k]*alpha*A[i, k] );

       but avoids recomputing alpha * A[i, k] and alpha * B[i, k] for
       every j and keeps C[i, j] in a register across the k-loop. */
    for (j = 0; j <= i; j++)
    {
      /* Fuse the scaling by beta with the rank-2k accumulation so that
         C[i][j] is loaded once, updated in a register, and stored once. */
      DATA_TYPE cij = C[i][j] * beta_local;

      /* Main k-reduction.  Access patterns:
           - A[j][k], B[j][k]: contiguous along k (row-major).
           - alphaA[k], alphaB[k]: contiguous along k.
         This is regular and well-suited for auto-vectorization. */
#pragma GCC ivdep
      for (k = 0; k < _PB_M; k++)
      {
        cij += A[j][k] * alphaB[k] + B[j][k] * alphaA[k];
      }

      C[i][j] = cij;
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syr2k (n, m,
		alpha, beta,
		POLYBENCH_ARRAY(C),
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}