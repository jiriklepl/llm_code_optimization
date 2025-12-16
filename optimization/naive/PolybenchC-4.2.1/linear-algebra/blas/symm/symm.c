/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <= i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; // regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   BLAS-like operation:

     SIDE = 'L'
     UPLO = 'L'

     C := alpha * A * B + beta * C

   where:
     - A is M x M, symmetric, stored in its lower triangle.
     - B is M x N.
     - C is M x N.

   The original code matched the reference BLAS algorithm but did:
     C[i][j] = beta * C[i][j] + ...
   inside the main loops, which creates a read-after-write dependence
   on C and inhibits vectorization.

   The optimized version below:
     1. Pre-scales C once: C := beta * C.
     2. Replaces the in-kernel "beta * C[i][j]" by a simple increment.
     3. Hoists loop invariants out of the inner loop.
     4. Applies simple (i,j) blocking to improve cache locality.

   All transformations preserve the mathematical result; the order of
   operations for each (i,j,k) triple is unchanged, so per-element
   accumulation order in k is identical to the original.
*/
static void __attribute__((flatten, noinline))
kernel_symm(int m, int n,
            DATA_TYPE alpha,
            DATA_TYPE beta,
            DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
            DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
            DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /* Local aliases for PolyBench loop bounds.
     These are usually equal to the runtime parameters m and n. */
  const int MB = _PB_M;
  const int NB = _PB_N;

#pragma scop

  /* --------------------------------------------------------------
   * Step 1: Pre-scale C by beta (C := beta * C).
   *
   * This allows us to remove the "beta * C[i][j]" term from the
   * main kernel body and simply add the newly computed value to
   * the already scaled C. This:
   *   - removes a multiply and a read-after-write dependency on C
   *   - makes the main loop friendlier to vectorization.
   * -------------------------------------------------------------- */
  for (i = 0; i < MB; ++i)
    for (j = 0; j < NB; ++j)
      C[i][j] *= beta;

  /* --------------------------------------------------------------
   * Step 2: Main symmetric matrix-matrix multiplication:
   *
   *   C := alpha * A * B + C
   *
   * where C has already been scaled by beta above.
   *
   * We apply a simple 2D blocking over (i,j) to improve data
   * locality for B and C. The block sizes are modest and chosen
   * to fit well in L1/L2 caches on typical x86-64 hardware, while
   * remaining portable.
   *
   * Blocking is done as:
   *   for (ii) for (jj) for (i in block ii) for (j in block jj)
   *
   * This preserves the original iteration order over (i,j) for each
   * row i (only the grouping of iterations changes), and there are
   * no dependences across different j, so semantics are unchanged.
   * -------------------------------------------------------------- */

  const int Ti = 32; /* i-block size */
  const int Tj = 32; /* j-block size */

  for (int ii = 0; ii < MB; ii += Ti)
  {
    int i_max = ii + Ti;
    if (i_max > MB) i_max = MB;

    for (int jj = 0; jj < NB; jj += Tj)
    {
      int j_max = jj + Tj;
      if (j_max > NB) j_max = NB;

      for (i = ii; i < i_max; ++i)
      {
        for (j = jj; j < j_max; ++j)
        {
          /* temp2 is the inner-product accumulation over k for
             the current (i,j) pair. */
          DATA_TYPE temp2 = (DATA_TYPE)0;

          /* Hoist invariants out of the k-loop: B[i][j] and the
             multiplication by alpha are constant across k. */
          DATA_TYPE Bij      = B[i][j];
          DATA_TYPE alphaBij = alpha * Bij;

          /* k-loop: strictly lower part of row i of A.
             Only A[i][k] with k < i are used. */
#pragma GCC ivdep
          for (k = 0; k < i; ++k)
          {
            DATA_TYPE Aik = A[i][k];

            /* Update C[k][j] using symmetric contribution:
                 C[k][j] += alpha * B[i][j] * A[i][k]; */
            C[k][j] += alphaBij * Aik;

            /* Accumulate temp2:
                 temp2 += B[k][j] * A[i][k]; */
            temp2 += B[k][j] * Aik;
          }

          /* Diagonal and accumulated lower-triangular contribution.

             Originally:
               C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i]
                         + alpha*temp2;

             Since we have already applied C := beta*C globally,
             C[i][j] currently holds beta*C0[i][j] (plus any
             contributions from previous i's via C[k][j] updates,
             exactly as in the original). Therefore we can safely
             use '+=' instead of '=' and drop the extra beta factor:
          */
          C[i][j] += alphaBij * A[i][i] + alpha * temp2;
        }
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
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}