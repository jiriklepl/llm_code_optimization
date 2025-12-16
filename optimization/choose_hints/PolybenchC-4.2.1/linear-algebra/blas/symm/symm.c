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

/* Tunable blocking factor for the column (j) dimension.
 * Can be overridden at compile time, e.g., -DSYMM_JBLOCK=128.
 * This value should be chosen so that:
 *   2 * SYMM_JBLOCK * sizeof(DATA_TYPE)
 * comfortably fits in L1/L2 cache.
 */
#ifndef SYMM_JBLOCK
# define SYMM_JBLOCK 64
#endif


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
   including the call and return.

   Optimizations applied:
   - Use of restrict-qualified local pointers for better alias analysis.
   - Blocking in the column (j) dimension to improve cache locality.
   - Reordering of loops from (i,j,k) to (j_block, i, k, j-in-block) to:
       * keep accesses to C, B contiguous in memory (row-major order),
       * reuse alpha * B(i,j) across all k for a fixed (i,j),
       * accumulate temp2(i,j) for all j in a block at once.
   - OpenMP parallelization over column blocks (each block is independent).
   - Inner j-loops are annotated with 'omp simd' to encourage vectorization.
   The mathematical operations performed for each (i,j,k) triple are
   unchanged: we still compute
       C[k][j] += alpha * B[i][j] * A[i][k]
       temp2(i,j) += B[k][j] * A[i][k]
       C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2(i,j)
   but in a cache- and vector-friendly order.
*/
static
void kernel_symm [[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  /* Local restrict-qualified views to help the compiler reason about aliasing.
   * A and B are read-only in the kernel; C is read-write.
   */
  DATA_TYPE       (*restrict C_)[n] = C;
  const DATA_TYPE (*restrict A_)[m] = A;
  const DATA_TYPE (*restrict B_)[n] = B;

  //BLAS PARAMS
  //SIDE = 'L'
  //UPLO = 'L'
  // =>  Form  C := alpha*A*B + beta*C
  // A is MxM (symmetric, lower triangular part stored/used)
  // B is MxN
  // C is MxN
  //note that due to Fortran array layout, the code below more closely
  //resembles upper triangular case in BLAS

#pragma scop
  /* We parallelize over blocks of columns (j dimension). Each block is
   * completely independent because different columns of C and B do not
   * interact. Blocking keeps inner loops over contiguous memory and
   * reduces cache misses.
   */
#pragma omp parallel for schedule(static)
  for (int jb = 0; jb < _PB_N; jb += SYMM_JBLOCK)
  {
    int j_end    = jb + SYMM_JBLOCK;
    if (j_end > _PB_N)
      j_end = _PB_N;
    int jb_width = j_end - jb; /* actual block width, handling the tail */

    /* Temporary buffers for this block and this thread:
     *  - alphaB_row[jb_width] = alpha * B[i][j] for j in [jb, j_end)
     *  - temp2[jb_width]     = sum_{k < i} B[k][j] * A[i][k]
     *
     * These are reused across all k for a given (block, i).
     */
    DATA_TYPE alphaB_row[jb_width];
    DATA_TYPE temp2[jb_width];

    /* Loop over rows of A and C (i dimension). We must keep this loop
     * sequential to respect the dependency:
     *   C[k][j] updated at iteration i depends on C[k][j] produced
     *   by all previous i' < i.
     */
    for (int i = 0; i < _PB_M; ++i)
    {
      const DATA_TYPE *restrict Bi_block = &B_[i][jb];

      /* Precompute alpha * B(i,j) for this block of columns and
       * initialize temp2 to zero.
       */
#pragma omp simd
      for (int jj = 0; jj < jb_width; ++jj)
      {
        alphaB_row[jj] = alpha * Bi_block[jj];
        temp2[jj]      = (DATA_TYPE)0;
      }

      /* Process strictly lower triangular part of A: k < i.
       * For each k, A[i][k] is reused across all j in the block, and
       * alphaB_row[j] is reused across all k for this (i,j), which
       * matches the original computation alpha * B[i][j] * A[i][k].
       */
      for (int k = 0; k < i; ++k)
      {
        const DATA_TYPE  aik       = A_[i][k];
        const DATA_TYPE *restrict Bk_block = &B_[k][jb];
        DATA_TYPE       *restrict Ck_block = &C_[k][jb];

#pragma omp simd
        for (int jj = 0; jj < jb_width; ++jj)
        {
          /* C[k][j] += alpha * B[i][j] * A[i][k]; */
          Ck_block[jj] += alphaB_row[jj] * aik;

          /* temp2(j) += B[k][j] * A[i][k]; */
          temp2[jj]   += Bk_block[jj] * aik;
        }
      }

      /* Diagonal contribution for row i of C.
       * For each j in the current block:
       *   C[i][j] = beta*C[i][j]
       *           + alpha*B[i][j]*A[i][i]
       *           + alpha*temp2(i,j)
       * We reuse alphaB_row[j] = alpha*B[i][j] here so that the
       * multiplication grouping ((alpha*B[i][j]) * A[i][i]) matches
       * the original code.
       */
      const DATA_TYPE aii = A_[i][i];
      DATA_TYPE *restrict Ci_block = &C_[i][jb];

#pragma omp simd
      for (int jj = 0; jj < jb_width; ++jj)
      {
        Ci_block[jj] = beta * Ci_block[jj]
                     + alphaB_row[jj] * aii
                     + alpha * temp2[jj];
      }
    } /* end loop over i */
  }   /* end loop over jb (column blocks) */
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