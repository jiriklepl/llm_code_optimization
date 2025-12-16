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
 *
 * Optimizations:
 * - Interchanged the (j,k) loops so that k is innermost. This makes
 *   all accesses to A, B and C in the innermost loop stride‑1 in
 *   row‑major storage, greatly improving data locality.
 * - Fused the beta scaling with the rank‑2k update so that each
 *   C[i][j] element is loaded and stored only once.
 * - Introduced row pointers (Ci, Ai, Bi, Aj, Bj) and a scalar
 *   accumulator (cij) to reduce repeated address calculations and
 *   enable better vectorization.
 * - The order of operations for each C[i][j] is preserved:
 *     C[i][j] is multiplied by beta, then contributions over k
 *     are accumulated in increasing k order, just as in the
 *     original implementation.
 */
static
void __attribute__((flatten, noinline))
kernel_syr2k(int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  /* Use the PolyBench loop bounds (may differ from n,m in some
     instrumentation modes). Cache them in locals to avoid repeated
     macro expansion and to give the compiler simple loop bounds. */
  const int ni = _PB_N;
  const int mi = _PB_M;

  for (i = 0; i < ni; i++) {
    /* Row pointers for faster address calculation and better
       vectorization. These do not change aliasing relationships:
       they merely provide convenient access to individual rows. */
    DATA_TYPE *restrict Ci = C[i];
    DATA_TYPE *restrict Ai = A[i];
    DATA_TYPE *restrict Bi = B[i];

    /* We now iterate j outside of k so that the innermost loop
       runs over k, which is the column index in row‑major
       A/B. For each (i,j), we:
         1) scale C[i][j] by beta
         2) accumulate contributions over k = 0..mi-1
       in that order, identically to the original kernel. */
    for (j = 0; j <= i; j++) {
      DATA_TYPE *restrict Aj = A[j];
      DATA_TYPE *restrict Bj = B[j];

      /* Load C[i][j], apply beta scaling once, and keep the
         accumulator in a scalar variable to avoid repeated
         memory traffic for C. */
      DATA_TYPE cij = Ci[j] * beta;

      /* Inner loop over k: fully contiguous accesses to all
         rows Ai, Bi, Aj, Bj (stride‑1), which is cache- and
         vectorization‑friendly. The algebraic form of the
         update is kept identical to the original:
           A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
         so as not to change floating‑point evaluation order. */
      for (k = 0; k < mi; k++) {
        const DATA_TYPE Ajk = Aj[k];
        const DATA_TYPE Bjk = Bj[k];
        const DATA_TYPE Aik = Ai[k];
        const DATA_TYPE Bik = Bi[k];

        cij += Ajk * alpha * Bik + Bjk * alpha * Aik;
      }

      /* Store the updated C[i][j] back to memory once. */
      Ci[j] = cij;
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