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

  /* Use local restrict-qualified pointers to help the optimizer and
     precompute 1/n as a DATA_TYPE to replace divisions by multiplies. */
  DATA_TYPE * restrict x1_ = x1;
  DATA_TYPE * restrict x2_ = x2;
  DATA_TYPE * restrict y1_ = y_1;
  DATA_TYPE * restrict y2_ = y_2;
  DATA_TYPE (* restrict A_)[n] = A;

  const DATA_TYPE inv_n = (DATA_TYPE)1.0 / (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      /* i % n == i for 0 <= i < n, but we keep the modulo in the
         expressions that wrap around (i+1), (i+3), (i+4) to preserve
         the original initialization pattern exactly. */
      x1_[i] = (DATA_TYPE)(i % n) * inv_n;

      int t = (i + 1) % n;
      x2_[i] = (DATA_TYPE)t * inv_n;

      t = (i + 3) % n;
      y1_[i] = (DATA_TYPE)t * inv_n;

      t = (i + 4) % n;
      y2_[i] = (DATA_TYPE)t * inv_n;

      /* Initialize A row by row, using multiplication by inv_n instead
         of division to reduce the cost of the operation. */
      DATA_TYPE * restrict Ai = A_[i];
      for (j = 0; j < n; j++)
      {
        int v = (i * j) % n;
        Ai[j] = (DATA_TYPE)v * inv_n;
      }
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

   Optimization notes:
   -------------------
   - The original code performed two separate matrix-vector products:
       x1 += A * y_1
       x2 += A^T * y_2
     in two passes over A, and updated x1[i]/x2[i] inside the inner loop.

   - Here we:
       * Fuse both operations into a single doubly-nested loop over A.
         Each A[i][j] is loaded once and contributes to both x1[i] and x2[j].
         This halves the total memory traffic to A and greatly improves
         data locality.
       * Accumulate x1[i] in a scalar register 'acc1' and write it back
         once per row, avoiding repeated loads/stores of x1[i].
       * Access A strictly row-major (A[i][j]) in the inner loop,
         which matches the storage layout.
       * Introduce local restrict-qualified pointers to let the compiler
         safely vectorize and reorder memory operations assuming no aliasing
         between the different arrays (which is guaranteed by PolyBench).

   - Importantly, the order of additions contributing to each element
     of x1 and x2 remains the same as in the original code:
       * For x1[i], contributions are still applied in strictly increasing j.
       * For x2[j], contributions are still applied in strictly increasing i.
     We only interleave operations on different indices, which does not
     change the final results.
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

  /* Create restrict-qualified local views to help the compiler with
     alias analysis and vectorization. These do not change semantics
     for the (non-overlapping) arrays used in PolyBench. */
  DATA_TYPE * restrict x1_ = x1;
  DATA_TYPE * restrict x2_ = x2;
  DATA_TYPE * restrict y1_ = y_1;
  DATA_TYPE * restrict y2_ = y_2;
  DATA_TYPE (* restrict A_)[n] = A;

#pragma scop
  for (i = 0; i < _PB_N; i++)
  {
    /* Accumulator for x1[i]; keeps the running sum in a register. */
    DATA_TYPE acc1 = x1_[i];

    /* y2[i] is invariant across the inner loop; load once. */
    const DATA_TYPE y2_i = y2_[i];

    /* Pointer to the current row of A for cache-friendly row-major access. */
    DATA_TYPE * restrict Ai = A_[i];

    /* Inner loop: for each column j, update:
         acc1   += A[i][j] * y1[j];
         x2[j]  += A[i][j] * y2[i];
       This computes:
         x1[i] = x1[i] + sum_j A[i][j] * y1[j]
         x2[j] = x2[j] + sum_i A[i][j] * y2[i]
       exactly as in the original two-loop version. */
    #pragma GCC ivdep
    for (j = 0; j < _PB_N; j++)
    {
      const DATA_TYPE A_ij = Ai[j];
      acc1      += A_ij * y1_[j];
      x2_[j]    += A_ij * y2_i;
    }

    x1_[i] = acc1;
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