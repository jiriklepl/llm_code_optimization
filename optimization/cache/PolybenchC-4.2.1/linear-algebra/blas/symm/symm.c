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

/* -------------------------------------------------------------------------
 * Tunable blocking parameter.
 *
 * SYMM_J_BLOCK controls the tile size along the N (column) dimension used
 * in the main computational kernel.  It can be adjusted at compile time
 * (e.g., -DSYMM_J_BLOCK=128) to better match the target cache hierarchy.
 * ------------------------------------------------------------------------- */
#ifndef SYMM_J_BLOCK
# define SYMM_J_BLOCK 64
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

  /* Initialize C and B: traverse rows (i) and write contiguous elements (j)
   * to improve cache locality. */
  for (i = 0; i < m; i++) {
    DATA_TYPE *Ci = C[i];
    DATA_TYPE *Bi = B[i];
    for (j = 0; j < n; j++) {
      Ci[j] = (DATA_TYPE) ((i + j) % 100) / m;
      Bi[j] = (DATA_TYPE) ((n + i - j) % 100) / m;
    }
  }

  /* Initialize A as lower-triangular; the upper part is filled with a
   * sentinel value (-999) and is never used by the kernel. */
  for (i = 0; i < m; i++) {
    DATA_TYPE *Ai = A[i];
    for (j = 0; j <= i; j++)
      Ai[j] = (DATA_TYPE) ((i + j) % 100) / m;
    for (j = i + 1; j < m; j++)
      Ai[j] = -999; /* regions of arrays that should not be used */
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

   Optimized algorithm description
   --------------------------------
   The original kernel computes (BLAS SYMM, side='L', uplo='L'):

       C := alpha * A * B + beta * C,

   where A is an M×M symmetric matrix stored in its lower triangular part,
   B is M×N, and C is M×N.

   Conceptually, let S be the full symmetric matrix such that:
       S[i][i] = A[i][i]
       S[i][k] = A[i][k] for k < i
       S[i][k] = A[k][i] for k > i   (using the stored lower triangle)

   Then:
       C_out = beta * C_in + alpha * S * B

   The original code implements this with an (i,j,k) loop nest and
   updates that access C[k][j] and B[k][j] with a large stride (column-wise
   in a row-major layout), which is cache- and vector-unfriendly.

   The optimized version below:
     1. First scales C by beta:  C[i][j] *= beta.
     2. Then explicitly computes the symmetric matrix-matrix product
        alpha * S * B and accumulates it into C, using only the lower
        triangular part of A (A[i][k] with k <= i):
          - Diagonal contributions:   A[i][i] * B[i][*]
          - Off-diagonal pairs (i > k):
                C[i][*] += alpha * A[i][k] * B[k][*]
                C[k][*] += alpha * A[i][k] * B[i][*]

   This is mathematically equivalent to the original kernel but:
     * Accesses rows of B and C contiguously (j is innermost),
       which improves spatial locality and enables efficient SIMD
       vectorization by the compiler.
     * Avoids inner loops over the strided dimension of row-major
       arrays (no more C[k][j] with k as the innermost index).
     * Uses a tunable blocking in j (columns) to improve cache
       reuse for large N.

   NOTE: The reordering uses algebraic equivalence; floating-point
   round-off may differ slightly but the computed operation is
   still C := alpha*A*B + beta*C using A as a symmetric matrix stored
   in its lower triangular part.
*/
static
void kernel_symm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /* Local constant for column blocking; must be > 0. */
  const int j_block = SYMM_J_BLOCK;

#pragma scop
  /* -----------------------------------------------------------------------
   * Step 1: Scale C by beta.
   *
   * This explicitly performs:
   *     C[i][j] <- beta * C[i][j]
   * for all i,j in the active problem size (0.._PB_M-1, 0.._PB_N-1).
   * --------------------------------------------------------------------- */
  for (i = 0; i < _PB_M; ++i) {
    DATA_TYPE *Ci = C[i];

    for (int jj = 0; jj < _PB_N; jj += j_block) {
      int j_end = jj + j_block;
      if (j_end > _PB_N)
        j_end = _PB_N;

      for (j = jj; j < j_end; ++j) {
        Ci[j] *= beta;
      }
    }
  }

  /* -----------------------------------------------------------------------
   * Step 2: Symmetric matrix-matrix product contribution.
   *
   * Using only the lower triangular part of A (including diagonal),
   * we compute alpha * S * B and add it to C.
   *
   * For each row i:
   *   - Add diagonal contribution:    alpha * A[i][i] * B[i][*]
   *   - For each k < i (off-diagonal pair):
   *        C[i][*] += alpha * A[i][k] * B[k][*]
   *        C[k][*] += alpha * A[i][k] * B[i][*]
   *
   * All innermost loops iterate over j, ensuring contiguous access to
   * rows of B and C, which significantly improves data locality and
   * enables efficient SIMD vectorization.
   * --------------------------------------------------------------------- */
  for (i = 0; i < _PB_M; ++i) {
    DATA_TYPE *Ai = A[i];
    DATA_TYPE *Bi = B[i];
    DATA_TYPE *Ci = C[i];

    /* Diagonal contribution: C[i][j] += alpha * A[i][i] * B[i][j] */
    DATA_TYPE alpha_aii = alpha * Ai[i];

    for (int jj = 0; jj < _PB_N; jj += j_block) {
      int j_end = jj + j_block;
      if (j_end > _PB_N)
        j_end = _PB_N;

      for (j = jj; j < j_end; ++j) {
        Ci[j] += alpha_aii * Bi[j];
      }
    }

    /* Off-diagonal contributions: use A[i][k] for all k < i. */
    for (k = 0; k < i; ++k) {
      DATA_TYPE aik        = Ai[k];          /* A[i][k], k < i (lower part) */
      DATA_TYPE alpha_aik  = alpha * aik;

      DATA_TYPE *Bk = B[k];
      DATA_TYPE *Ck = C[k];

      /* For this (i,k) pair, update rows i and k across all columns j. */
      for (int jj = 0; jj < _PB_N; jj += j_block) {
        int j_end = jj + j_block;
        if (j_end > _PB_N)
          j_end = _PB_N;

        for (j = jj; j < j_end; ++j) {
          DATA_TYPE Bi_j = Bi[j];
          DATA_TYPE Bk_j = Bk[j];

          /* C[i][j] += alpha * A[i][k] * B[k][j] */
          Ci[j] += alpha_aik * Bk_j;

          /* C[k][j] += alpha * A[i][k] * B[i][j] */
          Ck[j] += alpha_aik * Bi_j;
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