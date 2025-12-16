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

/* --------------------------------------------------------------------
 * Tunable blocking factor for columns of B in kernel_trmm.
 * This improves cache locality and enables efficient SIMD
 * on the innermost loops over columns of B.
 * Adjust this based on the target machine's cache line size
 * and SIMD width. 64 works well on modern x86-64.
 * ------------------------------------------------------------------ */
#ifndef TRMM_B_COL_BLOCK
#define TRMM_B_COL_BLOCK 64
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
   including the call and return.

   Original specification (BLAS-like):
   SIDE   = 'L'
   UPLO   = 'L'
   TRANSA = 'T'
   DIAG   = 'U'
   => Form  B := alpha * A**T * B.
   A is MxM (unit lower triangular)
   B is MxN

   Optimization strategy:
   - Preserve the mathematical computation:
       B[i][j] = alpha * ( B[i][j] +
                           sum_{k=i+1..M-1} A[k][i] * B[k][j] )
   - Improve data locality and SIMD efficiency by:
       * Blocking the computation along the column dimension (j)
         so that we operate on small tiles of B[i][*] kept in a
         stack-allocated buffer.
       * For each row i and tile of columns, we:
           1) Load B[i][j] for the tile into a temporary buffer.
           2) Accumulate contributions from all k>i into that buffer
              while accessing B[k][j] contiguously.
           3) Write the updated tile back to B[i][j].
       * Perform the global scaling by alpha in a separate pass over B.
         This is algebraically equivalent to the original kernel,
         because all uses of B[k][j] as inputs occur before any scaling
         of those entries, and the order of the additions for each
         B[i][j] is unchanged.
   - The temporary buffer has fixed size TRMM_B_COL_BLOCK, which is
     small (e.g., 64 doubles ~ 512 bytes) and well within the allowed
     memory overhead (< 50% of the original footprint).
*/
static
void kernel_trmm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /* Local restricted aliases to help the compiler with alias analysis
     and enable better vectorization. These have the same layout as
     the original PolyBench arrays but are marked restrict. */
  DATA_TYPE (*restrict A_)[m] = A;
  DATA_TYPE (*restrict B_)[n] = B;

  const int M_ = m;
  const int N_ = n;

  /* We separate the operation into two phases:

     1) Triangular multiplication without the final scaling:
          B[i][j] += A[k][i] * B[k][j]  for k > i

        This preserves the exact order of additions for each (i,j)
        as in the original implementation (k increases from i+1 to M_-1).

     2) Global scaling:
          B[i][j] *= alpha

        In the original code, each row i is scaled immediately after
        its updates, but that scaling happens after the last use of
        B[i][j] as a source term in any other row. Therefore, all
        the per-row scalings can be safely moved to the end without
        changing the numerical result (each element is still multiplied
        by alpha exactly once, after all additions).
  */

#pragma scop
  /* Phase 1: Compute B := B + strictly-lower(A^T) * B.
     Implemented in a cache- and SIMD-friendly, j-blocked fashion. */
  for (i = 0; i < M_; i++) {

    /* Process columns of B[i][*] in tiles of size TRMM_B_COL_BLOCK
       to increase locality and exploit SIMD along the column dimension. */
    for (int jb = 0; jb < N_; jb += TRMM_B_COL_BLOCK) {
      int j_end = jb + TRMM_B_COL_BLOCK;
      if (j_end > N_)
        j_end = N_;
      int tile_width = j_end - jb;

      /* Temporary buffer for the current row i and column tile [jb, j_end).
         The size is fixed at compile time to keep the array on the stack
         and avoid VLAs, but we use only the first tile_width elements. */
      DATA_TYPE Bi_tile[TRMM_B_COL_BLOCK];

      /* Pointer to the beginning of the destination row tile. */
      DATA_TYPE *restrict Bi_row_tile = &B_[i][jb];

      /* Load the initial values of B[i][j] for this tile into the buffer.
         This allows us to keep partial sums in fast local storage instead
         of repeatedly loading/storing B[i][j] from main memory. */
      #pragma GCC ivdep
      for (int jj = 0; jj < tile_width; jj++) {
        Bi_tile[jj] = Bi_row_tile[jj];
      }

      /* Accumulate contributions from rows k > i.
         For each k, A_[k][i] is reused across the tile, and we access
         B[k][jb..j_end-1] contiguously, which is good for caches and SIMD. */
      for (k = i + 1; k < M_; k++) {
        const DATA_TYPE aik = A_[k][i];
        const DATA_TYPE *restrict Bk_row_tile = &B_[k][jb];

        #pragma GCC ivdep
        for (int jj = 0; jj < tile_width; jj++) {
          Bi_tile[jj] += aik * Bk_row_tile[jj];
        }
      }

      /* Store the updated tile back to B[i][j]. */
      #pragma GCC ivdep
      for (int jj = 0; jj < tile_width; jj++) {
        Bi_row_tile[jj] = Bi_tile[jj];
      }
    }
  }

  /* Phase 2: Scale B by alpha: B := alpha * B.
     This loop is completely independent across (i,j) and is therefore
     trivially vectorizable and parallelizable by the compiler. */
  if (alpha != (DATA_TYPE)1.0) {
    for (i = 0; i < M_; i++) {
      #pragma GCC ivdep
      for (j = 0; j < N_; j++) {
        B_[i][j] *= alpha;
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