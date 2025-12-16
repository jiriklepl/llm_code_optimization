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
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
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
   including the call and return. */
static
void kernel_symm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /*
   * Optimized version of the original kernel.
   *
   * Original structure (high level, for reference):
   *
   *   for (i)
   *     for (j) {
   *       temp2 = 0;
   *       for (k < i) {
   *         C[k][j] += alpha * B[i][j] * A[i][k];
   *         temp2   +=         B[k][j] * A[i][k];
   *       }
   *       C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2;
   *     }
   *
   * Key transformations:
   *
   * 1. Change loop ordering from (i,j,k) to (i,k,j) for the inner
   *    update, and lift the “temp2” scalar into a temporary vector
   *    temp2[j]. For each fixed i we compute:
   *
   *      temp2[j] = sum_{k < i} B[k][j] * A[i][k]
   *
   *    exactly as in the original kernel, but we accumulate all columns
   *    in parallel inside the innermost j-loop.
   *
   * 2. With the (i,k,j) ordering, the j-loop becomes innermost:
   *      - C[k][j], B[i][j], B[k][j], and temp2[j] are accessed with
   *        unit stride along j, which improves cache locality and
   *        enables efficient SIMD vectorization.
   *
   * 3. Use small, row-local pointers marked restrict in the inner loops
   *    to aid the compiler’s alias analysis and vectorization.
   *
   * 4. Use #pragma omp simd on the innermost j-loops to encourage
   *    explicit SIMD vectorization (harmless if OpenMP is not enabled).
   *
   * The mathematical result is unchanged: every product
   *   B[i][j] * A[i][k] and B[k][j] * A[i][k]
   * is formed exactly once for the same (i,j,k) triplets as before,
   * and each C[i][j] still sees its original beta * C[i][j] scaling
   * followed by the same accumulated symmetric contributions.
   */

  /* Per-row accumulation buffer:
   * temp2[j] corresponds to the scalar "temp2" in the original kernel
   * for the current row i and column j.
   *
   * Size uses PolyBench's runtime-bound macro _PB_N, so it stays
   * consistent with the loop bounds.
   */
  DATA_TYPE temp2[_PB_N];

  //BLAS PARAMS
  //SIDE = 'L'
  //UPLO = 'L'
  // =>  Form  C := alpha*A*B + beta*C
  // A is MxM
  // B is MxN
  // C is MxN
  //note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
#pragma scop
  for (i = 0; i < _PB_M; i++)
  {
    /* Initialize the per-column accumulator for this row i. */
    for (j = 0; j < _PB_N; j++)
      temp2[j] = (DATA_TYPE)0;

    /* Pointer to the current row of B reused across all k. */
    DATA_TYPE *restrict Bi = &B[i][0];

    /* Process the strictly lower part A[i][k] for k < i.
     *
     * For each k, we:
     *   - update row k of C: C[k][j] += alpha * A[i][k] * B[i][j]
     *   - accumulate temp2[j] += A[i][k] * B[k][j]
     *
     * This is exactly the work of the original inner k-loop, but with
     * j as the innermost dimension for better locality and vectorization.
     */
    for (k = 0; k < i; k++)
    {
      const DATA_TYPE a_ik       = A[i][k];
      const DATA_TYPE a_ik_alpha = alpha * a_ik;

      DATA_TYPE *restrict Ck = &C[k][0];
      DATA_TYPE *restrict Bk = &B[k][0];

      /* Vectorizable, unit-stride loop over columns j. */
#pragma omp simd
      for (j = 0; j < _PB_N; j++)
      {
        Ck[j]     += a_ik_alpha * Bi[j];
        temp2[j]  += a_ik       * Bk[j];
      }
    }

    /* Diagonal contribution for row i. */
    const DATA_TYPE a_ii = A[i][i];

    DATA_TYPE *restrict Ci = &C[i][0];

    /* Final update for row i of C.
     *
     * This corresponds exactly to the original:
     *   C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2
     * where temp2 is now temp2[j].
     */
#pragma omp simd
    for (j = 0; j < _PB_N; j++)
    {
      Ci[j] = beta * Ci[j]
            + alpha * Bi[j] * a_ii
            + alpha * temp2[j];
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