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
 * Tunable blocking factors for the spatial dimensions.
 * These can be overridden at compile time, e.g.:
 *   -DHEAT3D_BLOCK_I=32 -DHEAT3D_BLOCK_J=32
 *
 * They are chosen to improve cache locality by working on tiles
 * of the 3D domain instead of full rows at once.
 * --------------------------------------------------------------------
 */
#ifndef HEAT3D_BLOCK_I
# define HEAT3D_BLOCK_I 32
#endif

#ifndef HEAT3D_BLOCK_J
# define HEAT3D_BLOCK_J 32
#endif


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
        A[i][j][k] = B[i][j][k] =
          (DATA_TYPE) (i + j + (n-k)) * 10 / (n);
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

   Optimizations applied:
   - Use the runtime parameters (tsteps, n) directly for loop bounds.
   - Spatial blocking in the i and j dimensions (HEAT3D_BLOCK_I/J)
     to improve cache locality.
   - OpenMP parallelization of the spatial loops (if compiled with
     -fopenmp; otherwise pragmas are ignored by the compiler).
   - Reduction of redundant memory accesses by caching the center
     point and precomputing 2 * center once per lattice point.
   These transformations preserve the mathematical computation
   performed at each lattice point and the overall update order:
     for each time step: B = stencil(A); then A = stencil(B).
*/
static __attribute__((noinline))
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  const int T = tsteps;
  const int N = n;

  /* Constants used in the stencil. Declared once to avoid
     re-materialization and to make their role explicit. */
  const DATA_TYPE c0_125 = SCALAR_VAL(0.125);
  const DATA_TYPE c2_0   = SCALAR_VAL(2.0);

  /* Guard against very small problem sizes that would make the
     interior (1 .. N-2) empty or invalid. PolyBench normally uses
     reasonably large N, so this is mostly defensive. */
  if (N < 3 || T <= 0)
    return;

#pragma scop
  for (int t = 0; t < T; ++t) {

    /* ----------------------------------------------------------------
     * First sweep: B = stencil(A)
     * Each interior point B[i][j][k] depends only on neighbors in A,
     * so the computation over the spatial domain is embarrassingly
     * parallel and can be tiled.
     * ----------------------------------------------------------------
     */
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 1; ii < N - 1; ii += HEAT3D_BLOCK_I) {
      for (int jj = 1; jj < N - 1; jj += HEAT3D_BLOCK_J) {

        const int i_end = (ii + HEAT3D_BLOCK_I < N - 1) ?
                          (ii + HEAT3D_BLOCK_I) : (N - 1);
        const int j_end = (jj + HEAT3D_BLOCK_J < N - 1) ?
                          (jj + HEAT3D_BLOCK_J) : (N - 1);

        for (int i = ii; i < i_end; ++i) {
          for (int j = jj; j < j_end; ++j) {

            /* Cache row pointers for faster and more regular
               memory access inside the k-loop. This does not
               change arithmetic order, only where values are
               loaded from. */
            DATA_TYPE       *restrict Bij   = &B[i][j][0];
            const DATA_TYPE *restrict Aij   = &A[i][j][0];
            const DATA_TYPE *restrict Aip1j = &A[i+1][j][0];
            const DATA_TYPE *restrict Aim1j = &A[i-1][j][0];
            const DATA_TYPE *restrict Aijp1 = &A[i][j+1][0];
            const DATA_TYPE *restrict Aijm1 = &A[i][j-1][0];

            for (int k = 1; k < N - 1; ++k) {
              const DATA_TYPE center     = Aij[k];
              const DATA_TYPE two_center = c2_0 * center;

              /* Original expression:
               *   0.125*(A[i+1][j][k] - 2*A[i][j][k] + A[i-1][j][k]) +
               *   0.125*(A[i][j+1][k] - 2*A[i][j][k] + A[i][j-1][k]) +
               *   0.125*(A[i][j][k+1] - 2*A[i][j][k] + A[i][j][k-1]) +
               *   A[i][j][k];
               *
               * We reuse "center" and "two_center" but preserve the
               * same grouping of operations and constants.
               */
              Bij[k] =
                  c0_125 * (Aip1j[k] - two_center + Aim1j[k])
                + c0_125 * (Aijp1[k] - two_center + Aijm1[k])
                + c0_125 * (Aij[k+1] - two_center + Aij[k-1])
                + center;
            }
          }
        }
      }
    }

    /* ----------------------------------------------------------------
     * Second sweep: A = stencil(B)
     * Same pattern as above, but with A and B swapped.
     * ----------------------------------------------------------------
     */
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 1; ii < N - 1; ii += HEAT3D_BLOCK_I) {
      for (int jj = 1; jj < N - 1; jj += HEAT3D_BLOCK_J) {

        const int i_end = (ii + HEAT3D_BLOCK_I < N - 1) ?
                          (ii + HEAT3D_BLOCK_I) : (N - 1);
        const int j_end = (jj + HEAT3D_BLOCK_J < N - 1) ?
                          (jj + HEAT3D_BLOCK_J) : (N - 1);

        for (int i = ii; i < i_end; ++i) {
          for (int j = jj; j < j_end; ++j) {

            DATA_TYPE       *restrict Aij   = &A[i][j][0];
            const DATA_TYPE *restrict Bij   = &B[i][j][0];
            const DATA_TYPE *restrict Bip1j = &B[i+1][j][0];
            const DATA_TYPE *restrict Bim1j = &B[i-1][j][0];
            const DATA_TYPE *restrict Bijp1 = &B[i][j+1][0];
            const DATA_TYPE *restrict Bijm1 = &B[i][j-1][0];

            for (int k = 1; k < N - 1; ++k) {
              const DATA_TYPE center     = Bij[k];
              const DATA_TYPE two_center = c2_0 * center;

              Aij[k] =
                  c0_125 * (Bip1j[k] - two_center + Bim1j[k])
                + c0_125 * (Bijp1[k] - two_center + Bijm1[k])
                + c0_125 * (Bij[k+1] - two_center + Bij[k-1])
                + center;
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