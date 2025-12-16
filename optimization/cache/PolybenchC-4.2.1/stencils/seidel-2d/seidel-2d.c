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
   - Uses restrict-qualified row pointers for better alias analysis.
   - Implements a 3x3 sliding window along the j dimension to keep
     the stencil neighborhood in registers and minimize memory loads.
   - Preserves the original in-place Gauss–Seidel update order exactly.
*/
static
void __attribute__((flatten, noinline))
kernel_seidel_2d(int tsteps,
                 int n,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  /* Prevent unused-parameter warnings; PolyBench provides its own
     sanitized loop bounds via the _PB_* macros. */
  (void)tsteps;
  (void)n;

  /* Use PolyBench's “sanitized” problem sizes. */
  const int T = _PB_TSTEPS;
  const int N_eff = _PB_N;

  /* Create a restrict-qualified view of the 2D array.  This tells the
     compiler that accesses through A_ do not alias any other pointer,
     enabling better optimization. */
  DATA_TYPE (*restrict A_)[n] = A;

  const DATA_TYPE divisor = SCALAR_VAL(9.0);

  int t, i;

#pragma scop
  for (t = 0; t <= T - 1; ++t)
  {
    /* Sweep rows in the same order as the original code. */
    for (i = 1; i <= N_eff - 2; ++i)
    {
      /* Row pointers for the (i-1), i, and (i+1) rows.  Mark them as
         restrict because each points to a distinct row. */
      DATA_TYPE *restrict row_im1 = A_[i - 1];
      DATA_TYPE *restrict row_i   = A_[i];
      DATA_TYPE *restrict row_ip1 = A_[i + 1];

      const int j_start = 1;
      const int j_end   = N_eff - 2;

      /* If there is no interior column, skip (should not occur for
         standard PolyBench sizes, but keeps the code robust). */
      if (j_end < j_start)
        continue;

      int j = j_start;

      /* --- Initialize sliding 3x3 window for j == 1 -----------------
         At each iteration, the window holds the values:
           row_im1: (j-1, j, j+1)
           row_i  : (j-1, j, j+1)
           row_ip1: (j-1, j, j+1)
         These correspond exactly to the 3x3 stencil neighborhood
         used in the original code for A[i][j].
      */
      DATA_TYPE im1_jm1 = row_im1[j - 1];
      DATA_TYPE im1_j   = row_im1[j    ];
      DATA_TYPE im1_jp1 = row_im1[j + 1];

      DATA_TYPE cur_jm1 = row_i  [j - 1];
      DATA_TYPE cur_j   = row_i  [j    ];
      DATA_TYPE cur_jp1 = row_i  [j + 1];

      DATA_TYPE ip1_jm1 = row_ip1[j - 1];
      DATA_TYPE ip1_j   = row_ip1[j    ];
      DATA_TYPE ip1_jp1 = row_ip1[j + 1];

      /* Main loop for all interior columns except the last one:
         j runs from 1 to j_end - 1 (if j_end > 1).
         After each iteration we slide the window one column to the right
         by reusing most registers and only loading the new rightmost
         column (j+2) from memory.  This preserves the original
         lexicographic Gauss–Seidel update order:
           - row i-1: already fully updated in this time step
           - row i  : updated from left to right (dependencies in j)
           - row i+1: still old values from previous time step
      */
      if (j_end > j_start)
      {
        for (; j < j_end; ++j)
        {
          /* Compute the 3x3 average from the window. */
          DATA_TYPE sum =
              im1_jm1 + im1_j + im1_jp1 +
              cur_jm1 + cur_j + cur_jp1 +
              ip1_jm1 + ip1_j + ip1_jp1;

          DATA_TYPE new_center = sum / divisor;

          /* In-place Gauss–Seidel update, identical to A[i][j] = ... */
          row_i[j] = new_center;
          cur_j    = new_center;  /* updated value used in next iteration */

          /* Advance the sliding window by one column:
             - shift (j, j+1) to become (j-1, j)
             - load the new (j+1) == (j+2 of previous iteration) */
          im1_jm1 = im1_j;
          im1_j   = im1_jp1;

          cur_jm1 = cur_j;   /* new_center becomes left neighbor */

          ip1_jm1 = ip1_j;
          ip1_j   = ip1_jp1;

          const int jp2 = j + 2;
          im1_jp1 = row_im1[jp2];
          cur_jp1 = row_i  [jp2];
          ip1_jp1 = row_ip1[jp2];
        }
      }

      /* Handle the last interior column j == j_end.
         At this point, the window already corresponds to
         (j_end-1, j_end, j_end+1) for each of the three rows. */
      {
        DATA_TYPE sum =
            im1_jm1 + im1_j + im1_jp1 +
            cur_jm1 + cur_j + cur_jp1 +
            ip1_jm1 + ip1_j + ip1_jp1;

        DATA_TYPE new_center = sum / divisor;
        row_i[j] = new_center;
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