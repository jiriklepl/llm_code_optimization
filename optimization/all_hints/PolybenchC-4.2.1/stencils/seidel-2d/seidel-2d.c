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
   - Uses row pointers (Ai_1, Ai, Ai1) to reduce address computation.
   - Implements a 3x3 sliding window along the j dimension. For each
     new j position, 6 of the 9 stencil values are reused from the
     previous position and only 3 new elements are loaded from memory.
     This keeps the exact Gauss-Seidel update order (lexicographic in
     (i,j)) but substantially reduces memory traffic and improves use
     of registers and caches.
   - Division by 9.0 is kept (via c9) to preserve the original
     arithmetic structure as closely as possible.
*/
static
void kernel_seidel_2d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  /* Use PolyBench-provided loop bounds so behavior matches the
     original kernel exactly (they may differ from raw tsteps/n). */
  const int T  = _PB_TSTEPS;
  const int N_ = _PB_N;

  /* For very small grids there is no interior point to update.
     The original loops would not execute in that case. */
  if (N_ <= 2 || T <= 0)
    return;

  const DATA_TYPE c9 = SCALAR_VAL(9.0);

  int t, i, j;

#pragma scop
  for (t = 0; t < T; ++t)
  {
    for (i = 1; i <= N_ - 2; ++i)
    {
      /* Row base pointers. Rows are disjoint, so the 'restrict'
         qualifiers are valid and enable better optimization. */
      DATA_TYPE * restrict Ai_1 = A[i-1];
      DATA_TYPE * restrict Ai   = A[i];
      DATA_TYPE * restrict Ai1  = A[i+1];

      /* Initialize a 3x3 sliding window centered at j = 1.
         Indices mirror the original stencil:
           im1_* : row i-1 (already updated for time t)
           i_*   : row i   (being updated)
           ip1_* : row i+1 (not yet updated for time t) */
      DATA_TYPE im1_jm1 = Ai_1[0];
      DATA_TYPE im1_j   = Ai_1[1];
      DATA_TYPE im1_jp1 = Ai_1[2];

      DATA_TYPE  i_jm1  = Ai[0];
      DATA_TYPE  i_j    = Ai[1];
      DATA_TYPE  i_jp1  = Ai[2];

      DATA_TYPE ip1_jm1 = Ai1[0];
      DATA_TYPE ip1_j   = Ai1[1];
      DATA_TYPE ip1_jp1 = Ai1[2];

      for (j = 1; j <= N_ - 2; ++j)
      {
        /* Compute new value for A[i][j] using the current window.
           This matches the original 9-point stencil exactly. */
        DATA_TYPE sum =
          im1_jm1 + im1_j + im1_jp1 +
           i_jm1  +  i_j  +  i_jp1  +
          ip1_jm1 + ip1_j + ip1_jp1;

        DATA_TYPE new_val = sum / c9;
        Ai[j] = new_val;

        /* Advance the sliding window one column to the right.
           The update strictly follows the original Gauss-Seidel
           ordering:
             - A[i][j-1] (left neighbor) becomes 'new_val'
             - A[i][j]   and A[i][j+1] remain the yet-unmodified
               values for the next iteration.
           This way every stencil uses exactly the same neighbor
           values (old vs. newly updated) as the original code. */
        if (j < N_ - 2)
        {
          const int jp2 = j + 2;

          /* Row i-1 (fully updated for this time step). */
          im1_jm1 = im1_j;
          im1_j   = im1_jp1;
          im1_jp1 = Ai_1[jp2];

          /* Row i (currently being updated). */
          i_jm1   = new_val;   /* new A[i][j] becomes left neighbor */
          i_j     = i_jp1;     /* old A[i][j+1] becomes center      */
          i_jp1   = Ai[jp2];   /* old A[i][j+2] becomes right       */

          /* Row i+1 (not yet updated for this time step). */
          ip1_jm1 = ip1_j;
          ip1_j   = ip1_jp1;
          ip1_jp1 = Ai1[jp2];
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