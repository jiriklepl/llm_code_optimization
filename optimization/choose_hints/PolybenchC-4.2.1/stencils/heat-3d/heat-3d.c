/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations:
   - Use pointers to 1D k-lines (A[i][j], B[i][j]) to avoid repeated
     3D address calculation and to improve cache locality and SIMD
     vectorization along k.
   - Hoist loop-invariant constants (0.125 and 2.0) out of the loops.
   - Reuse the center value and its 2x multiple to reduce redundant
     loads and multiplications.
   - The mathematical stencil is unchanged: this still implements the
     7-point 3D heat diffusion update, with the same interior/boundary
     behavior as the original code.
*/
static
void kernel_heat_3d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  /* The original PolyBench kernels drive the iteration space using the
     TSTEPS/_PB_N macros rather than the function parameters.  The
     cast-to-void keeps this behavior explicit and avoids warnings. */
  (void)tsteps;
  (void)n;

  /* Interior region: indices in [1, _PB_N-2]. */
  const int istart = 1;
  const int iend   = _PB_N - 1; /* exclusive upper bound */

  /* Precompute scalar constants once. */
  const DATA_TYPE c0 = SCALAR_VAL(0.125);
  const DATA_TYPE c2 = SCALAR_VAL(2.0);

#pragma scop
  for (int t = 1; t <= TSTEPS; t++)
  {
    /*--------------------------------------------------------------
      First sweep: update B from A (A -> B).
      Each updated point depends only on values from A at time t-1,
      so there are no write-after-read or write-after-write
      dependencies between different (i,j,k) points in this loop.
    --------------------------------------------------------------*/
#ifdef _OPENMP
    /* Parallelize over i,j if OpenMP is enabled.  Without -fopenmp,
       this pragma is ignored by the compiler. */
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = istart; i < iend; i++)
    {
      for (int j = istart; j < iend; j++)
      {
        /* Pointers to k-lines for the current (i,j) and its 6 neighbors.
           They are marked restrict because distinct (i,j) pairs map to
           disjoint k-lines in memory. This helps the compiler to
           safely reorder and vectorize loads/stores along k. */
        DATA_TYPE * restrict a_center = A[i][j];
        DATA_TYPE * restrict a_im1    = A[i-1][j];
        DATA_TYPE * restrict a_ip1    = A[i+1][j];
        DATA_TYPE * restrict a_jm1    = A[i][j-1];
        DATA_TYPE * restrict a_jp1    = A[i][j+1];
        DATA_TYPE * restrict b_center = B[i][j];

        for (int k = istart; k < iend; k++)
        {
          const DATA_TYPE center  = a_center[k];
          const DATA_TYPE center2 = c2 * center; /* 2.0 * center */

          /* These three sums correspond exactly to the three
             parenthesized terms in the original code, one per
             spatial dimension. Factoring out the common
             (2.0 * center) is algebraically equivalent. */
          const DATA_TYPE sum_i =
              a_ip1[k]      /* A[i+1][j][k] */
            - center2
            + a_im1[k];     /* A[i-1][j][k] */

          const DATA_TYPE sum_j =
              a_jp1[k]      /* A[i][j+1][k] */
            - center2
            + a_jm1[k];     /* A[i][j-1][k] */

          const DATA_TYPE sum_k =
              a_center[k+1] /* A[i][j][k+1] */
            - center2
            + a_center[k-1];/* A[i][j][k-1] */

          /* Equivalent to:
             B = 0.125*sum_i + 0.125*sum_j + 0.125*sum_k + center; */
          b_center[k] = center + c0 * (sum_i + sum_j + sum_k);
        }
      }
    }

    /*--------------------------------------------------------------
      Second sweep: update A from B (B -> A).
      Same stencil, but with A and B swapped.
    --------------------------------------------------------------*/
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = istart; i < iend; i++)
    {
      for (int j = istart; j < iend; j++)
      {
        DATA_TYPE * restrict b_center = B[i][j];
        DATA_TYPE * restrict b_im1    = B[i-1][j];
        DATA_TYPE * restrict b_ip1    = B[i+1][j];
        DATA_TYPE * restrict b_jm1    = B[i][j-1];
        DATA_TYPE * restrict b_jp1    = B[i][j+1];
        DATA_TYPE * restrict a_center = A[i][j];

        for (int k = istart; k < iend; k++)
        {
          const DATA_TYPE center  = b_center[k];
          const DATA_TYPE center2 = c2 * center; /* 2.0 * center */

          const DATA_TYPE sum_i =
              b_ip1[k]      /* B[i+1][j][k] */
            - center2
            + b_im1[k];     /* B[i-1][j][k] */

          const DATA_TYPE sum_j =
              b_jp1[k]      /* B[i][j+1][k] */
            - center2
            + b_jm1[k];     /* B[i][j-1][k] */

          const DATA_TYPE sum_k =
              b_center[k+1] /* B[i][j][k+1] */
            - center2
            + b_center[k-1];/* B[i][j][k-1] */

          a_center[k] = center + c0 * (sum_i + sum_j + sum_k);
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
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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