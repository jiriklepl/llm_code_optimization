/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"

/* ----------------------------------------------------------------------
 * Tunable blocking parameters for the main kernel.
 *
 * T_I: tile size in the row dimension  (i)
 * T_J: tile size in the column dimension (j) for the trailing panel
 *
 * These are chosen to get tiles that fit comfortably in L1/L2 cache.
 * They can be adjusted at compile time by defining T_I / T_J.
 * --------------------------------------------------------------------*/
#ifndef T_I
# define T_I 32
#endif

#ifndef T_J
# define T_J 32
#endif


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
   http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt[[gnu::flatten, gnu::noinline]](int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE nrm;

#pragma scop
  /* Outer loop over columns k is kept sequential: each step depends on
     the updated A and R from previous k. */
  for (k = 0; k < _PB_N; k++)
    {
      /* ----------------------------------------------------------------
       * 1) Compute squared 2-norm of column k of A:
       *        nrm = sum_i A[i][k]^2
       * ----------------------------------------------------------------*/
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
      {
        DATA_TYPE aik = A[i][k];
        nrm += aik * aik;
      }

      /* ----------------------------------------------------------------
       * 2) Compute R[k][k] = ||A[:,k]||_2 and normalize column k:
       *
       *    Original code:
       *      R[k][k] = sqrt(nrm);
       *      for i: Q[i][k] = A[i][k] / R[k][k];
       *
       *    Optimized:
       *      - compute rkk = sqrt(nrm);
       *      - compute inv_rkk = 1 / rkk once per k;
       *      - Q[i][k] = A[i][k] * inv_rkk;
       *
       *    This replaces M divisions per k with one division and M
       *    multiplications (strength reduction), preserving algebraic
       *    results.
       * ----------------------------------------------------------------*/
      DATA_TYPE rkk     = SQRT_FUN(nrm);
      DATA_TYPE inv_rkk = SCALAR_VAL(1.0) / rkk;
      R[k][k] = rkk;

      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] * inv_rkk;

      /* ----------------------------------------------------------------
       * 3) Initialize the k-th row of R for columns j > k:
       *        R[k][j] = 0
       * ----------------------------------------------------------------*/
      for (j = k + 1; j < _PB_N; j++)
        R[k][j] = SCALAR_VAL(0.0);

      /* ----------------------------------------------------------------
       * 4) Build row k of R for j > k:
       *
       *      R[k][j] = sum_i Q[i][k] * A[i][j]
       *
       *    Original code:
       *      for j = k+1..N-1:
       *        R[k][j] = 0;
       *        for i = 0..M-1:
       *          R[k][j] += Q[i][k] * A[i][j];
       *
       *    That loop order (j outer, i inner) walks A column-wise
       *    in the inner loop, which is poor for row-major storage.
       *
       *    Here we use i-outer / j-inner order and apply 2D blocking:
       *
       *      for ii in tiles of i:
       *        for jj in tiles of j>k:
       *          for i in ii-tile:
       *            qki = Q[i][k];              // one load per (k,i)
       *            for j in jj-tile:
       *              R[k][j] += qki * A[i][j]; // row-wise over A
       *
       *    Because R[k][j] is initialized to zero before this and
       *    each pair (i,j) is visited exactly once, the accumulation
       *    order for each R[k][j] remains the same over i, preserving
       *    the scalar reduction semantics while greatly improving
       *    cache locality on A (row-major, unit stride in j).
       * ----------------------------------------------------------------*/
      {
        /* Tile rows (i) and trailing columns (j) for better cache use. */
        for (int ii = 0; ii < _PB_M; ii += T_I)
        {
          int i_end = ii + T_I;
          if (i_end > _PB_M)
            i_end = _PB_M;

          for (int jj = k + 1; jj < _PB_N; jj += T_J)
          {
            int j_end = jj + T_J;
            if (j_end > _PB_N)
              j_end = _PB_N;

            for (i = ii; i < i_end; i++)
            {
              /* Q[i][k] is reused across all columns in this (k,i) row. */
              DATA_TYPE qki = Q[i][k];

              for (j = jj; j < j_end; j++)
              {
                R[k][j] += qki * A[i][j];
              }
            }
          }
        }
      }

      /* ----------------------------------------------------------------
       * 5) Orthogonalize trailing columns of A against Q[:,k]:
       *
       *      A[i][j] = A[i][j] - Q[i][k] * R[k][j]   for all j > k.
       *
       *    We use the same tiling and loop order as above:
       *
       *      for ii in tiles of i:
       *        for jj in tiles of j>k:
       *          for i in ii-tile:
       *            qki = Q[i][k];
       *            for j in jj-tile:
       *              A[i][j] -= qki * R[k][j];
       *
       *    This keeps accesses to A row-major and contiguous in j,
       *    and reuses qki across the inner loop.
       * ----------------------------------------------------------------*/
      {
        for (int ii = 0; ii < _PB_M; ii += T_I)
        {
          int i_end = ii + T_I;
          if (i_end > _PB_M)
            i_end = _PB_M;

          for (int jj = k + 1; jj < _PB_N; jj += T_J)
          {
            int j_end = jj + T_J;
            if (j_end > _PB_N)
              j_end = _PB_N;

            for (i = ii; i < i_end; i++)
            {
              DATA_TYPE qki = Q[i][k];

              for (j = jj; j < j_end; j++)
              {
                A[i][j] -= qki * R[k][j];
              }
            }
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
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}