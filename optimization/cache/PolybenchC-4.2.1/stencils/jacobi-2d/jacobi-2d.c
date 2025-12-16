/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* --------------------------------------------------------------------
 * Tunable blocking parameters for the stencil kernel.
 *
 * These control the tile sizes in the i- and j-dimensions.  They can
 * be overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DJACOBI_2D_BLOCK_I=64 -DJACOBI_2D_BLOCK_J=64 ...
 *
 * Reasonable defaults are chosen for modern x86-64 CPUs, but the best
 * values are hardware- and problem-size-dependent.
 * ------------------------------------------------------------------*/
#ifndef JACOBI_2D_BLOCK_I
#  define JACOBI_2D_BLOCK_I 32
#endif

#ifndef JACOBI_2D_BLOCK_J
#  define JACOBI_2D_BLOCK_J 32
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
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

   - Introduce restrict-qualified local aliases for A and B to remove
     potential aliasing and enable more aggressive vectorization.
   - Use row pointers (A_[i], B_[i]) to reduce 2D indexing overhead and
     ensure unit-stride access in the innermost loop.
   - Apply simple 2D blocking (tiling) over i and j using the tunable
     JACOBI_2D_BLOCK_I/J parameters to improve cache locality on large
     grids.
   - Hoist loop-invariant quantities (bounds and the scalar weight).
 */
static
void kernel_jacobi_2d[[gnu::flatten, gnu::noinline]](int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  /* Use PolyBench-controlled loop bounds inside the kernel, as in the
   * original version. These may differ from the raw parameters when
   * PolyBench instrumentation is enabled. */
  const int tsteps_ = _PB_TSTEPS;
  const int n_      = _PB_N;

  /* Local restrict-qualified aliases improve optimization opportunities.
   * We only use A_ and B_ below, never A or B directly, to honor
   * the 'restrict' contract (A_ and B_ point to non-overlapping arrays). */
  DATA_TYPE (*restrict A_)[n_] = A;
  DATA_TYPE (*restrict B_)[n_] = B;

  /* Stencil weight; hoisted to avoid repeated macro expansion and
   * constant folding inside the inner loops. */
  const DATA_TYPE w = SCALAR_VAL(0.2);

  /* Common loop bounds for the interior region (the boundary is fixed). */
  const int i_start = 1;
  const int i_end   = n_ - 1; /* i in [1, n_-2] */
  const int j_start = 1;
  const int j_end   = n_ - 1; /* j in [1, n_-2] */

#pragma scop
  for (int t = 0; t < tsteps_; ++t)
    {
      /* --------------------------------------------------------------
       * Phase 1: B = stencil(A)
       *
       * Each B[i][j] depends only on values from A, which is read-only
       * in this phase. There are no loop-carried dependences, so we are
       * free to reorder the (i,j) iterations and to tile them.
       * ----------------------------------------------------------- */
      for (int ii = i_start; ii < i_end; ii += JACOBI_2D_BLOCK_I)
      {
        /* Handle the last, potentially smaller, tile safely. */
        const int i_max = (ii + JACOBI_2D_BLOCK_I < i_end)
                          ? (ii + JACOBI_2D_BLOCK_I)
                          : i_end;

        for (int jj = j_start; jj < j_end; jj += JACOBI_2D_BLOCK_J)
        {
          const int j_max = (jj + JACOBI_2D_BLOCK_J < j_end)
                            ? (jj + JACOBI_2D_BLOCK_J)
                            : j_end;

          for (int i = ii; i < i_max; ++i)
          {
            /* Row pointers: Bi, Ai, Aim (A[i-1]), Aip (A[i+1]).
             * This transforms 2D indexing (A[i][j]) into 1D indexing
             * (Ai[j]), which is cheaper and clearly unit-stride in 'j'. */
            DATA_TYPE *restrict Bi  = B_[i];
            DATA_TYPE *restrict Ai  = A_[i];
            DATA_TYPE *restrict Aim = A_[i-1];
            DATA_TYPE *restrict Aip = A_[i+1];

            for (int j = jj; j < j_max; ++j)
            {
              Bi[j] = w * (Ai[j]     /* center   */
                         +  Ai[j-1]  /* left     */
                         +  Ai[j+1]  /* right    */
                         +  Aip[j]   /* down     */
                         +  Aim[j]); /* up       */
            }
          }
        }
      }

      /* --------------------------------------------------------------
       * Phase 2: A = stencil(B)
       *
       * Symmetric to phase 1: now A is updated from read-only B.
       * Again, there are no loop-carried dependences across (i,j).
       * ----------------------------------------------------------- */
      for (int ii = i_start; ii < i_end; ii += JACOBI_2D_BLOCK_I)
      {
        const int i_max = (ii + JACOBI_2D_BLOCK_I < i_end)
                          ? (ii + JACOBI_2D_BLOCK_I)
                          : i_end;

        for (int jj = j_start; jj < j_end; jj += JACOBI_2D_BLOCK_J)
        {
          const int j_max = (jj + JACOBI_2D_BLOCK_J < j_end)
                            ? (jj + JACOBI_2D_BLOCK_J)
                            : j_end;

          for (int i = ii; i < i_max; ++i)
          {
            DATA_TYPE *restrict Ai  = A_[i];
            DATA_TYPE *restrict Bi  = B_[i];
            DATA_TYPE *restrict Bim = B_[i-1];
            DATA_TYPE *restrict Bip = B_[i+1];

            for (int j = jj; j < j_max; ++j)
            {
              Ai[j] = w * (Bi[j]     /* center   */
                         +  Bi[j-1]  /* left     */
                         +  Bi[j+1]  /* right    */
                         +  Bip[j]   /* down     */
                         +  Bim[j]); /* up       */
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
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}