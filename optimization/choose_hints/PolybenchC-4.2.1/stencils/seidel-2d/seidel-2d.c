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

   Optimized version:
   ------------------
   The original kernel directly loaded all 9 stencil points from memory
   for every (i, j) update:

       A[i][j] = (9-point sum) / 9;

   This results in 9 memory loads per lattice point, even though
   adjacent points in the j-direction reuse many of the same values.

   The implementation below keeps a 3x3 sliding window of stencil values
   in registers while sweeping along j. For each new j, only the three
   "rightmost" values of the window (one from each of the three rows)
   are reloaded from memory; the other six are reused from registers.

   Semantics:
   ----------
   The loop order (t, then i increasing, then j increasing) and the
   in-place Gauss-Seidel update order are preserved exactly, so the
   numerical results are identical to the original code.
*/
static
void kernel_seidel_2d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

  /* Local restrict-qualified alias to help the compiler with alias
     analysis and enable more aggressive optimizations. */
  DATA_TYPE (*restrict a)[n] = A;

  /* Constant denominator used in the stencil. Hoisted out of loops to
     avoid re-materializing the SCALAR_VAL macro at every iteration. */
  const DATA_TYPE denom = SCALAR_VAL(9.0);

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
  {
    const int n_inner = _PB_N;
    const int i_start = 1;
    const int i_end   = n_inner - 2;
    const int j_start = 1;
    const int j_end   = n_inner - 2;

    for (i = i_start; i <= i_end; i++)
    {
      /* Row pointers for the 3x3 stencil around (i, j). Using row
         pointers reduces address arithmetic compared to repeated A[i][j]
         indexing and improves cache locality. */
      DATA_TYPE *restrict row_above = a[i-1];
      DATA_TYPE *restrict row_curr  = a[i];
      DATA_TYPE *restrict row_below = a[i+1];

      /* For very small problems with no interior points, the j loop in
         the original code would not execute. We retain that behavior. */
      if (j_end < j_start)
        continue;

      /* Initialize a sliding 3x3 stencil window for j = j_start (== 1).
         The window variables hold:
             top_im1_*  : row i-1
             mid_i_*    : row i
             bot_ip1_*  : row i+1
         with suffixes jm1, j, jp1 for columns j-1, j, j+1. */
      j = j_start;

      DATA_TYPE top_im1_jm1 = row_above[j-1];
      DATA_TYPE top_im1_j   = row_above[j  ];
      DATA_TYPE top_im1_jp1 = row_above[j+1];

      DATA_TYPE mid_i_jm1   = row_curr [j-1];
      DATA_TYPE mid_i_j     = row_curr [j  ];
      DATA_TYPE mid_i_jp1   = row_curr [j+1];

      DATA_TYPE bot_ip1_jm1 = row_below[j-1];
      DATA_TYPE bot_ip1_j   = row_below[j  ];
      DATA_TYPE bot_ip1_jp1 = row_below[j+1];

      for (; j <= j_end; j++)
      {
        /* Compute the 9-point average using the values currently in
           the window. This matches the original expression exactly. */
        DATA_TYPE sum =
            top_im1_jm1 + top_im1_j + top_im1_jp1 +
            mid_i_jm1   + mid_i_j   + mid_i_jp1   +
            bot_ip1_jm1 + bot_ip1_j + bot_ip1_jp1;

        DATA_TYPE new_center = sum / denom;

        /* In-place Gauss-Seidel update; same memory write as in the
           original kernel. */
        row_curr[j] = new_center;

        /* Slide the stencil window one step to the right, preparing
           data for the next j (j+1). This carefully preserves the
           original dependency structure:

             - top/bottom rows are read-only within a time step;
             - for the middle row, the left neighbor at j+1 must see
               the *updated* A[i][j], which we keep in 'new_center';
             - all other reused values remain from the previous time
               step, just as in the original code. */
        if (j < j_end)
        {
          const int jp2 = j + 2;  /* Column j+2 is the new rightmost one. */

          /* Shift previously loaded data:
               current center/right become next left/center. */
          top_im1_jm1 = top_im1_j;
          top_im1_j   = top_im1_jp1;

          mid_i_jm1   = new_center;   /* updated A[i][j] */
          mid_i_j     = mid_i_jp1;    /* old A[i][j+1]   */

          bot_ip1_jm1 = bot_ip1_j;
          bot_ip1_j   = bot_ip1_jp1;

          /* Load the new rightmost column (j+2) for all three rows.
             These are still "old" values for the current time step,
             exactly as required by the Gauss-Seidel scheme. */
          top_im1_jp1 = row_above[jp2];
          mid_i_jp1   = row_curr [jp2];
          bot_ip1_jp1 = row_below[jp2];
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