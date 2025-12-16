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

/* --------------------------------------------------------------------
 * Tunable blocking factors for the spatial loops.
 *
 * These control how the 3D domain is partitioned in the i/j dimensions
 * to improve cache locality.  They can be overridden at compile time,
 * e.g.:
 *   -DHEAT3D_BLOCK_I=32 -DHEAT3D_BLOCK_J=32
 * ------------------------------------------------------------------ */
#ifndef HEAT3D_BLOCK_I
# define HEAT3D_BLOCK_I 16
#endif

#ifndef HEAT3D_BLOCK_J
# define HEAT3D_BLOCK_J 16
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  /* Local pointer aliases make the subscript arithmetic explicit and
   * allow the compiler to better optimize and vectorize the loops. */
  DATA_TYPE (* __restrict A_)[n][n] = A;
  DATA_TYPE (* __restrict B_)[n][n] = B;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      DATA_TYPE * __restrict a_ij = &A_[i][j][0];
      DATA_TYPE * __restrict b_ij = &B_[i][j][0];

      /* k is the fastest varying dimension (contiguous in memory). */
      #pragma GCC ivdep
      for (k = 0; k < n; k++)
      {
        DATA_TYPE val = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
        a_ij[k] = val;
        b_ij[k] = val;
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
   including the call and return. */
static
void kernel_heat_3d[[gnu::flatten, gnu::noinline]](int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t;

  /* Use local pointer aliases with explicit dimensions.  This makes
   * the memory layout clearer to the compiler and enables better
   * optimization (including vectorization and strength reduction of
   * index arithmetic). */
  const int dim = n;
  DATA_TYPE (* __restrict A_)[dim][dim] = A;
  DATA_TYPE (* __restrict B_)[dim][dim] = B;

  /* Constants used in the stencil; hoisted out of inner loops. */
  const DATA_TYPE c0 = SCALAR_VAL(0.125);
  const DATA_TYPE c2 = SCALAR_VAL(2.0);

  /* PolyBench-provided "sanitized" loop bound for the spatial size. */
  const int n_inner = _PB_N;

  /* These are the maximum indices of the interior domain.
   * Original code iterates: for (i = 1; i < _PB_N-1; ++i)
   * so interior i,j,k are in [1, n_inner-2]. We keep exactly the same
   * iteration set, but we tile in i and j for better cache behavior. */
  const int i_max = n_inner - 1;
  const int j_max = n_inner - 1;
  const int k_max = n_inner - 1;

#pragma scop
  for (t = 1; t <= TSTEPS; t++) {

    /* ----------------------------------------------------------------
     * First sweep: update B from A
     *   B[i][j][k] <- f(A and its 6 neighbors)
     * ---------------------------------------------------------------- */
    {
      const int TI = HEAT3D_BLOCK_I;
      const int TJ = HEAT3D_BLOCK_J;

      for (int ii = 1; ii < i_max; ii += TI) {
        int i_end = ii + TI;
        if (i_end > i_max) i_end = i_max;   /* i in [1, i_max-1] */

        for (int jj = 1; jj < j_max; jj += TJ) {
          int j_end = jj + TJ;
          if (j_end > j_max) j_end = j_max; /* j in [1, j_max-1] */

          for (int i = ii; i < i_end; ++i) {
            for (int j = jj; j < j_end; ++j) {

              /* Pointers to the current line along k and its
               * six neighbors.  k is the fastest dimension, so
               * these pointers allow unit-stride access in k. */
              DATA_TYPE       * __restrict b     = &B_[i][j][0];
              const DATA_TYPE * __restrict a     = &A_[i][j][0];
              const DATA_TYPE * __restrict a_ip1 = &A_[i+1][j][0];
              const DATA_TYPE * __restrict a_im1 = &A_[i-1][j][0];
              const DATA_TYPE * __restrict a_jp1 = &A_[i][j+1][0];
              const DATA_TYPE * __restrict a_jm1 = &A_[i][j-1][0];

              /* Inner-most loop: contiguous in memory, ideal for SIMD. */
              #pragma GCC ivdep
              for (int k = 1; k < k_max; ++k) {
                const DATA_TYPE center      = a[k];
                const DATA_TYPE two_center  = c2 * center;

                /* Algebraically identical to the original expression:
                 *
                 *   0.125 * (A[i+1] - 2*A[i] + A[i-1]) +
                 *   0.125 * (A[i][j+1] - 2*A[i] + A[i][j-1]) +
                 *   0.125 * (A[i][k+1] - 2*A[i] + A[i][k-1]) +
                 *   A[i]
                 *
                 * We only hoist the repeated factor (2 * center);
                 * the grouping of terms and operations is preserved. */
                b[k] = c0 * (a_ip1[k] - two_center + a_im1[k])
                     + c0 * (a_jp1[k] - two_center + a_jm1[k])
                     + c0 * (a[k+1]   - two_center + a[k-1])
                     + center;
              }
            }
          }
        }
      }
    }

    /* ----------------------------------------------------------------
     * Second sweep: update A from B
     *   A[i][j][k] <- f(B and its 6 neighbors)
     * The loop structure is identical to the first sweep, just with
     * the roles of A and B swapped.
     * ---------------------------------------------------------------- */
    {
      const int TI = HEAT3D_BLOCK_I;
      const int TJ = HEAT3D_BLOCK_J;

      for (int ii = 1; ii < i_max; ii += TI) {
        int i_end = ii + TI;
        if (i_end > i_max) i_end = i_max;

        for (int jj = 1; jj < j_max; jj += TJ) {
          int j_end = jj + TJ;
          if (j_end > j_max) j_end = j_max;

          for (int i = ii; i < i_end; ++i) {
            for (int j = jj; j < j_end; ++j) {

              DATA_TYPE       * __restrict a     = &A_[i][j][0];
              const DATA_TYPE * __restrict b     = &B_[i][j][0];
              const DATA_TYPE * __restrict b_ip1 = &B_[i+1][j][0];
              const DATA_TYPE * __restrict b_im1 = &B_[i-1][j][0];
              const DATA_TYPE * __restrict b_jp1 = &B_[i][j+1][0];
              const DATA_TYPE * __restrict b_jm1 = &B_[i][j-1][0];

              #pragma GCC ivdep
              for (int k = 1; k < k_max; ++k) {
                const DATA_TYPE center      = b[k];
                const DATA_TYPE two_center  = c2 * center;

                a[k] = c0 * (b_ip1[k] - two_center + b_im1[k])
                     + c0 * (b_jp1[k] - two_center + b_jm1[k])
                     + c0 * (b[k+1]   - two_center + b[k-1])
                     + center;
              }
            }
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