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
 * Tunable parameters
 * --------------------------------------------------------------------
 *
 * MVT_BLOCK_SIZE controls a simple blocking factor for the inner
 * dimension (the j-loop) of the matrix-vector products.  It can be
 * overridden at compile time, e.g.:
 *
 *   -DMVT_BLOCK_SIZE=128
 *
 * to experiment with different cache behaviors on a given machine.
 */
#ifndef MVT_BLOCK_SIZE
# define MVT_BLOCK_SIZE 64
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
   including the call and return.
 *
 * Optimizations applied:
 *  - Use a scalar accumulator per row/column (x1[i], x2[i]) to keep
 *    these values in registers across the inner loop, instead of
 *    repeatedly loading and storing them on every iteration.
 *    This preserves the exact floating-point evaluation order
 *    (j increases from 0 to _PB_N-1) for each x1[i] and x2[i].
 *
 *  - Simple blocking over the inner j dimension (MVT_BLOCK_SIZE)
 *    to improve locality for y_1 / y_2 and for the rows/columns of A.
 *    Blocking is done in a way that preserves the per-element update
 *    order (still in increasing j) and handles non-multiple sizes.
 *
 *  - OpenMP parallelization over the outer i-loop for both
 *    matrix-vector products. Each i-iteration writes a distinct
 *    x1[i] or x2[i], so there are no data races.  The two
 *    matrix-vector products are placed in a single parallel region
 *    to amortize thread creation/destruction overhead.
 *
 *  - The access pattern for A in the first product (A[i][j]) is
 *    row-major and therefore cache-friendly.  In the second product
 *    (A[j][i]) the access pattern is inherently column-wise; we keep
 *    the same indexing to preserve the mathematical structure, but
 *    the scalar accumulator still reduces memory traffic for x2.
 */
static
void kernel_mvt[[gnu::flatten, gnu::noinline]](int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;
  const int n_ = _PB_N;

#pragma scop
  /* Create a single parallel region and use two work-sharing loops
     inside it. This avoids paying the cost of entering/exiting a
     parallel region twice. */
#pragma omp parallel
  {
    /* ------------------------------------------------------------
     * First matrix-vector product: x1 = x1 + A * y_1
     * ------------------------------------------------------------
     *
     * Original form:
     *   for (i = 0; i < _PB_N; i++)
     *     for (j = 0; j < _PB_N; j++)
     *       x1[i] = x1[i] + A[i][j] * y_1[j];
     *
     * Here we:
     *   - parallelize on i (each iteration is independent),
     *   - accumulate into a scalar 'sum' to keep x1[i] in a register,
     *   - apply simple blocking on j using MVT_BLOCK_SIZE.
     */
#pragma omp for schedule(static)
    for (i = 0; i < n_; i++)
    {
      DATA_TYPE sum = x1[i];

      /* Cache a pointer to row i of A so the compiler can generate
         efficient base+offset addressing. */
      DATA_TYPE* Ai = A[i];

      /* Block the inner dimension to improve cache locality. */
      for (int jj = 0; jj < n_; jj += MVT_BLOCK_SIZE)
      {
        int j_end = jj + MVT_BLOCK_SIZE;
        if (j_end > n_)
          j_end = n_;

        /* The order of j (0..n_-1) is preserved overall, so the
           floating-point accumulation order for each x1[i] is
           identical to the original code. */
        for (j = jj; j < j_end; j++)
        {
          sum += Ai[j] * y_1[j];
        }
      }

      x1[i] = sum;
    }

    /* ------------------------------------------------------------
     * Second matrix-vector product: x2 = x2 + A^T * y_2
     * ------------------------------------------------------------
     *
     * Original form:
     *   for (i = 0; i < _PB_N; i++)
     *     for (j = 0; j < _PB_N; j++)
     *       x2[i] = x2[i] + A[j][i] * y_2[j];
     *
     * We apply the same style of optimization as above:
     *   - parallelize on i,
     *   - keep x2[i] in a register via 'sum',
     *   - block over j while preserving the order j=0..n_-1
     *     for each x2[i].
     */
#pragma omp for schedule(static)
    for (i = 0; i < n_; i++)
    {
      DATA_TYPE sum = x2[i];

      for (int jj = 0; jj < n_; jj += MVT_BLOCK_SIZE)
      {
        int j_end = jj + MVT_BLOCK_SIZE;
        if (j_end > n_)
          j_end = n_;

        for (j = jj; j < j_end; j++)
        {
          sum += A[j][i] * y_2[j];
        }
      }

      x2[i] = sum;
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