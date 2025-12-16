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

  /* Use a restrict-qualified pointer and per-row pointers to
     help the compiler generate better code and reduce the
     cost of repeated 2D indexing. */
  DATA_TYPE (*restrict A_)[n] = A;

  for (i = 0; i < n; i++)
  {
    DATA_TYPE *restrict row = A_[i];
    for (j = 0; j < n; j++)
      row[j] = ((DATA_TYPE) i*(j+2) + 2) / n;
  }
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

   Optimizations applied:
   - Use a restrict-qualified local alias for A to improve alias analysis.
   - Strip-mine (block) the i and j loops to work on smaller pieces of
     the grid at a time, improving cache locality for large N.
   - Introduce per-row pointers (row_im1, row_i, row_ip1) to avoid
     repeated 2D index calculations.
   - Manually unroll the innermost j loop by a factor of 4 while keeping
     the original lexicographic (i,j) traversal order. This is crucial
     to preserve the Gauss-Seidel update semantics (each point sees the
     same mix of updated and non-updated neighbors as in the original
     code).
*/
static
void kernel_seidel_2d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

  /* Local restrict-qualified alias to help the compiler.
     The original pointer A is not used after this point. */
  DATA_TYPE (*restrict A_)[n] = A;

  const int tmax   = _PB_TSTEPS;
  const int istart = 1;
  const int iend   = _PB_N - 2;
  const int jstart = 1;
  const int jend   = _PB_N - 2;

  /* Blocking factors. They are chosen to be small enough to improve
     cache locality but large enough to avoid excessive loop overhead.
     These values are heuristic and work well on modern x86-64 cores. */
  const int I_BLOCK = 32;
  const int J_BLOCK = 64;

#pragma scop
  for (t = 0; t <= tmax - 1; ++t)
  {
    /* Strip-mine the i-dimension.  The (i,j) pairs are still visited
       in lexicographic order: rows are processed in increasing i,
       and within each row j increases from 1 to jend. */
    for (int ii = istart; ii <= iend; ii += I_BLOCK)
    {
      const int i_max = (ii + I_BLOCK - 1 < iend) ? (ii + I_BLOCK - 1) : iend;

      for (i = ii; i <= i_max; ++i)
      {
        /* Cache row pointers for the three rows used by the stencil. */
        DATA_TYPE *restrict row_im1 = A_[i-1];
        DATA_TYPE *restrict row_i   = A_[i];
        DATA_TYPE *restrict row_ip1 = A_[i+1];

        /* Strip-mine the j-dimension inside each row.  The overall
           order along j is still strictly increasing, so the dependence
           on A[i][j-1] (Gauss-Seidel behavior along the row) is
           preserved exactly. */
        for (int jj = jstart; jj <= jend; jj += J_BLOCK)
        {
          const int j_max = (jj + J_BLOCK - 1 < jend) ? (jj + J_BLOCK - 1) : jend;

          /* Manually unroll the innermost loop by 4. This reduces loop
             control overhead without changing the effective execution
             order of the individual updates. */
          for (j = jj; j + 3 <= j_max; j += 4)
          {
            int j0 = j;
            row_i[j0] =
              (row_im1[j0-1] + row_im1[j0] + row_im1[j0+1]
               + row_i[j0-1] + row_i[j0] + row_i[j0+1]
               + row_ip1[j0-1] + row_ip1[j0] + row_ip1[j0+1])
              / SCALAR_VAL(9.0);

            int j1 = j0 + 1;
            row_i[j1] =
              (row_im1[j1-1] + row_im1[j1] + row_im1[j1+1]
               + row_i[j1-1] + row_i[j1] + row_i[j1+1]
               + row_ip1[j1-1] + row_ip1[j1] + row_ip1[j1+1])
              / SCALAR_VAL(9.0);

            int j2 = j0 + 2;
            row_i[j2] =
              (row_im1[j2-1] + row_im1[j2] + row_im1[j2+1]
               + row_i[j2-1] + row_i[j2] + row_i[j2+1]
               + row_ip1[j2-1] + row_ip1[j2] + row_ip1[j2+1])
              / SCALAR_VAL(9.0);

            int j3 = j0 + 3;
            row_i[j3] =
              (row_im1[j3-1] + row_im1[j3] + row_im1[j3+1]
               + row_i[j3-1] + row_i[j3] + row_i[j3+1]
               + row_ip1[j3-1] + row_ip1[j3] + row_ip1[j3+1])
              / SCALAR_VAL(9.0);
          }

          /* Remainder loop for any columns not covered by the unrolled loop. */
          for (; j <= j_max; ++j)
          {
            row_i[j] =
              (row_im1[j-1] + row_im1[j] + row_im1[j+1]
               + row_i[j-1] + row_i[j] + row_i[j+1]
               + row_ip1[j-1] + row_ip1[j] + row_ip1[j+1])
              / SCALAR_VAL(9.0);
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