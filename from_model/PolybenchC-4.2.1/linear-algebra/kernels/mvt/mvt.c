/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"

/* --------------------------------------------------------------------
 * Tunable tiling parameters for the optimized kernel.
 *
 * MVT_TILE_I:
 *   - Number of rows of A (and elements of x1 / y_2) processed together.
 *   - Controls the size of the row tile.
 *
 * MVT_TILE_J:
 *   - Number of columns of A (and elements of x2 / y_1) processed together.
 *   - Controls the size of the column tile.
 *
 * The default values are chosen so that a MVT_TILE_I × MVT_TILE_J block
 * of A, plus the corresponding pieces of the vectors, fits comfortably
 * in the L1/L2 caches of a modern x86-64 CPU.
 * ------------------------------------------------------------------ */
#ifndef MVT_TILE_I
#  define MVT_TILE_I 32
#endif

#ifndef MVT_TILE_J
#  define MVT_TILE_J 256
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt[[gnu::flatten, gnu::noinline]](int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  /* We use a fused, tiled formulation of the original two matrix–vector
   * products:
   *
   *   Original kernel:
   *     // x1 = x1 + A * y_1
   *     for (i)
   *       for (j)
   *         x1[i] += A[i][j] * y_1[j];
   *
   *     // x2 = x2 + A^T * y_2
   *     for (i)
   *       for (j)
   *         x2[i] += A[j][i] * y_2[j];
   *
   * Algebraic rewrite of the second product:
   *   x2[i] = x2[i] + sum_j A[j][i] * y_2[j]
   *         = x2[i] + sum_k A[k][i] * y_2[k]          (rename j->k)
   *   or, equivalently (renaming i<->j for the outer index):
   *   x2[j] = x2[j] + sum_i A[i][j] * y_2[i]
   *
   * After this index renaming, both products use A[i][j] in row-major
   * order. We can then fuse them into:
   *
   *   for (i)
   *     for (j)
   *       x1[i] += A[i][j] * y_1[j];
   *       x2[j] += A[i][j] * y_2[i];
   *
   * This fused form:
   *   - halves the number of loads of A[i][j];
   *   - preserves the mathematical result of the original two-loop
   *     sequence (up to floating-point round-off).
   *
   * Below we additionally:
   *   - tile in both i and j (MVT_TILE_I × MVT_TILE_J blocks),
   *   - keep per-row accumulators for x1 in registers across j-tiles,
   *   - accumulate x2 updates in a small temporary tile before adding
   *     them to the global x2, reducing global memory traffic.
   */

#pragma scop
  const int tile_i = MVT_TILE_I;
  const int tile_j = MVT_TILE_J;

  for (int I = 0; I < _PB_N; I += tile_i)
  {
    const int i_end = (I + tile_i < _PB_N) ? (I + tile_i) : _PB_N;
    const int tile_height = i_end - I;

    /* Accumulators for x1 over this I-tile:
     *   x1_acc[ii] holds the current value for row i = I + ii,
     *   including contributions from all J-tiles processed so far.
     */
    DATA_TYPE x1_acc[MVT_TILE_I];
    /* Cached y_2 entries for this tile; y_2 is read-only, so we can
     * load each element once per row tile and reuse it.
     */
    DATA_TYPE y2_tile[MVT_TILE_I];

    for (int ii = 0; ii < tile_height; ++ii)
    {
      const int row = I + ii;
      x1_acc[ii]   = x1[row];
      y2_tile[ii]  = y_2[row];
    }

    /* Loop over column tiles. */
    for (int J = 0; J < _PB_N; J += tile_j)
    {
      const int j_end = (J + tile_j < _PB_N) ? (J + tile_j) : _PB_N;
      const int tile_width = j_end - J;

      /* Local accumulator for x2 over this (I,J) tile:
       *   x2_tile[jj] accumulates contributions to x2[col] for all
       *   rows row in the current I-tile, and columns
       *   col = J + jj in the current J-tile.
       *
       * Using this buffer dramatically reduces the number of loads
       * and stores to the global x2 array: each x2 element is updated
       * once per I-tile instead of once per row.
       */
      DATA_TYPE x2_tile[MVT_TILE_J];

      /* Initialize partial sums for this tile. */
      for (int jj = 0; jj < tile_width; ++jj)
        x2_tile[jj] = (DATA_TYPE)0;

      /* Process all rows in the current I-tile for this column tile. */
      for (int ii = 0; ii < tile_height; ++ii)
      {
        const int row = I + ii;
        DATA_TYPE acc1      = x1_acc[ii];   /* accumulator for x1[row] */
        const DATA_TYPE y2i = y2_tile[ii];  /* y_2[row], reused across j */

        /* Pointer to the beginning of the current sub-row of A. */
        const DATA_TYPE *A_row = &A[row][J];

        /* Inner-most loop:
         *   - A_row[jj] and y_1[col] are accessed contiguously,
         *   - x2_tile[jj] is a small local array, also accessed
         *     contiguously.
         * This pattern is well-suited for auto-vectorization.
         */
        for (int jj = 0; jj < tile_width; ++jj)
        {
          const int col       = J + jj;
          const DATA_TYPE aij = A_row[jj];

          acc1          += aij * y_1[col];  /* contribution to x1[row]   */
          x2_tile[jj]   += aij * y2i;       /* contribution to x2[col]   */
        }

        x1_acc[ii] = acc1;
      }

      /* Accumulate this tile's contributions into the global x2. */
      for (int jj = 0; jj < tile_width; ++jj)
      {
        const int col = J + jj;
        x2[col] += x2_tile[jj];
      }
    }

    /* Write back the accumulated x1 values for this I-tile. */
    for (int ii = 0; ii < tile_height; ++ii)
    {
      const int row = I + ii;
      x1[row] = x1_acc[ii];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}