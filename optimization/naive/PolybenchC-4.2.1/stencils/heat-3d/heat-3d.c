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


/* Array initialization.
 *
 * Optimization:
 * - Keep k as the innermost loop (contiguous dimension).
 * - Use row pointers so the compiler performs less address arithmetic
 *   in the innermost loop (better data locality on the k dimension).
 * - Compute the value once per element, then store it to both A and B.
 */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      /* Pointers to the contiguous k-lines for this (i,j). */
      DATA_TYPE *restrict A_ij = &A[i][j][0];
      DATA_TYPE *restrict B_ij = &B[i][j][0];

      for (k = 0; k < n; k++)
      {
        /* Keep arithmetic identical to the original expression:
           (DATA_TYPE) (i + j + (n-k)) * 10 / n
           Only hoisted into a temporary for reuse. */
        DATA_TYPE v = (DATA_TYPE) (i + j + (n-k)) * 10 / n;
        A_ij[k] = v;
        B_ij[k] = v;
      }
    }
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
 *
 * Optimizations applied:
 * - Hoist constant scalar coefficients out of the time loop.
 * - For each (i,j), create restrict-qualified pointers to the k-lines of
 *   A and B and of their 6 neighbors. This reduces multi-dimensional
 *   index computations in the innermost loop and improves cache usage.
 * - Reuse the value A[i][j][k] (or B[i][j][k]) and the product 2*center
 *   across all three spatial directions. This removes redundant loads
 *   and multiplies while preserving the exact evaluation order of
 *   additions and subtractions within each term.
 * - Add GCC-specific hints for vectorization and unrolling on the
 *   innermost (k) loop, which is free of loop-carried dependencies.
 *
 * The algebraic structure of the stencil is kept identical to the
 * original code (no reassociation of floating-point additions), so
 * numerical behaviour remains consistent with the reference version.
 */
static
void __attribute__((flatten, noinline))
kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;

  /* Hoisted scalar constants (computed once per kernel call). */
  const DATA_TYPE c0 = SCALAR_VAL(0.125);
  const DATA_TYPE c2 = SCALAR_VAL(2.0);

#pragma scop
  for (t = 1; t <= TSTEPS; t++) {
    /* First sweep: update B from A. */
    for (i = 1; i < _PB_N-1; i++) {
      for (j = 1; j < _PB_N-1; j++) {
        /* Cache pointers to the 7 relevant k-lines for this (i,j):
           center and its +/-1 neighbors along i and j.
           Each of these is a contiguous segment in memory. */
        DATA_TYPE *restrict A_ij    = &A[i  ][j  ][0];
        DATA_TYPE *restrict A_ip1j  = &A[i+1][j  ][0];
        DATA_TYPE *restrict A_im1j  = &A[i-1][j  ][0];
        DATA_TYPE *restrict A_ijp1  = &A[i  ][j+1][0];
        DATA_TYPE *restrict A_ijm1  = &A[i  ][j-1][0];
        DATA_TYPE *restrict B_ij    = &B[i  ][j  ][0];

        /* k is the fastest-varying dimension; encourage the compiler
           to vectorize and aggressively unroll this loop. */
#pragma GCC ivdep
#pragma GCC unroll 8
        for (k = 1; k < _PB_N-1; k++) {
          DATA_TYPE center     = A_ij[k];
          DATA_TYPE two_center = c2 * center;

          /* Each term_x/term_y/term_z corresponds exactly to the
             original directional contributions, only expressed using
             cached pointers and the precomputed 2*center. */
          DATA_TYPE term_x =
            c0 * (A_ip1j[k] - two_center + A_im1j[k]);
          DATA_TYPE term_y =
            c0 * (A_ijp1[k] - two_center + A_ijm1[k]);
          DATA_TYPE term_z =
            c0 * (A_ij[k+1] - two_center + A_ij[k-1]);

          B_ij[k] = term_x + term_y + term_z + center;
        }
      }
    }

    /* Second sweep: update A from B (symmetric to the first sweep). */
    for (i = 1; i < _PB_N-1; i++) {
      for (j = 1; j < _PB_N-1; j++) {
        DATA_TYPE *restrict B_ij    = &B[i  ][j  ][0];
        DATA_TYPE *restrict B_ip1j  = &B[i+1][j  ][0];
        DATA_TYPE *restrict B_im1j  = &B[i-1][j  ][0];
        DATA_TYPE *restrict B_ijp1  = &B[i  ][j+1][0];
        DATA_TYPE *restrict B_ijm1  = &B[i  ][j-1][0];
        DATA_TYPE *restrict A_ij    = &A[i  ][j  ][0];

#pragma GCC ivdep
#pragma GCC unroll 8
        for (k = 1; k < _PB_N-1; k++) {
          DATA_TYPE center     = B_ij[k];
          DATA_TYPE two_center = c2 * center;

          DATA_TYPE term_x =
            c0 * (B_ip1j[k] - two_center + B_im1j[k]);
          DATA_TYPE term_y =
            c0 * (B_ijp1[k] - two_center + B_ijm1[k]);
          DATA_TYPE term_z =
            c0 * (B_ij[k+1] - two_center + B_ij[k-1]);

          A_ij[k] = term_x + term_y + term_z + center;
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