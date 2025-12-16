/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied while preserving Gauss–Seidel semantics:
   - Strength reduction: replace division by 9.0 with multiplication by a
     precomputed reciprocal 'inv9'.
   - 1D blocking in the i (row) dimension to improve cache/TLB locality,
     without changing the global (t,i,j) execution order.
   - Sliding 3-point window along j for the rows i-1 and i+1. This reuses
     partial sums and reduces memory traffic from 9 loads per grid point
     to ~5, while keeping the exact in-place Gauss–Seidel update order:
       * row i-1: uses values already updated at this time step,
       * row i:   mixes newly updated A[i][j-1] with old A[i][j],A[i][j+1],
       * row i+1: uses values not yet updated at this time step.
   - No parallelization or reordering that would break the lexicographic
     (t, i, j) order is introduced.
*/
static
void kernel_seidel_2d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

  /* Precompute 1/9 once per kernel call (strength reduction). */
  const DATA_TYPE inv9 = SCALAR_VAL(1.0) / SCALAR_VAL(9.0);

  /* Tile size for the i dimension. This groups rows into bands while
     preserving their global order. A small multiple of cache lines is
     a reasonable choice on typical x64 machines. */
  const int tile_i = 32;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
  {
    /* Strip-mine / block the i-loop.
       Order of (i,j) pairs over the whole grid is unchanged:
         original: for i = 1..N-2, for j = 1..N-2
         blocked : for ii, for i = ii.., for j = 1..N-2
       with ii increasing. */
    for (int ii = 1; ii <= _PB_N - 2; ii += tile_i)
    {
      const int i_end = (ii + tile_i - 1 <= _PB_N - 2)
                        ? (ii + tile_i - 1)
                        : (_PB_N - 2);

      for (i = ii; i <= i_end; i++)
      {
        /* Row pointers for faster and clearer access. Each points to a
           distinct row, so they do not alias each other. */
        DATA_TYPE *restrict Ai_minus1 = A[i-1];
        DATA_TYPE *restrict Ai        = A[i];
        DATA_TYPE *restrict Ai_plus1  = A[i+1];

        /* Initialize sliding 3-point horizontal sums for j = 1.
           We use columns 0,1,2 in rows (i-1), i, and (i+1). */
        DATA_TYPE s_up  = Ai_minus1[0] + Ai_minus1[1] + Ai_minus1[2];
        DATA_TYPE s_mid = Ai       [0] + Ai       [1] + Ai       [2];
        DATA_TYPE s_dn  = Ai_plus1 [0] + Ai_plus1 [1] + Ai_plus1 [2];

        /* First interior column: j = 1.
           Uses:
             - row i-1:   cols 0,1,2 (already updated this time step),
             - row i:     cols 0,1,2 (all from previous time step),
             - row i+1:   cols 0,1,2 (not yet updated at this time step). */
        Ai[1] = (s_up + s_mid + s_dn) * inv9;

        /* Main sweep along the row: j = 2 .. N-2.
           For each step we slide the 3-wide window one cell to the right
           for rows i-1 and i+1 and recompute the middle-row sum so that:
             - A[i, j-1] is the newly updated value,
             - A[i, j]   and A[i, j+1] are still old values. */
        for (j = 2; j <= _PB_N - 2; j++)
        {
          /* Slide windows on rows i-1 and i+1:
               old window: [j-2, j-1, j]
               new window: [j-1, j,   j+1] */
          s_up += Ai_minus1[j+1] - Ai_minus1[j-2];
          s_dn += Ai_plus1 [j+1] - Ai_plus1 [j-2];

          /* Central row window at current (t,i,j):
             A[i, j-1] has just been updated at this time step,
             A[i, j] and A[i, j+1] are still from the previous step. */
          s_mid = Ai[j-1] + Ai[j] + Ai[j+1];

          Ai[j] = (s_up + s_mid + s_dn) * inv9;
        }
      }
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d (tsteps, n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}