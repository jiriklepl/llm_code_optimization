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

/* -----------------------------------------------------------------
 * Tunable spatial tiling parameters for the Jacobi kernel.
 *
 * These control the i/j tile sizes used inside kernel_jacobi_2d.
 * They are chosen to keep a tile of A and B (plus 1-cell halo)
 * comfortably in L1/L2 cache. You may experiment with other values.
 * ----------------------------------------------------------------- */
#ifndef JACOBI_2D_TILE_I
# define JACOBI_2D_TILE_I 32
#endif

#ifndef JACOBI_2D_TILE_J
# define JACOBI_2D_TILE_J 32
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
   including the call and return. */
static
void kernel_jacobi_2d[[gnu::flatten, gnu::noinline]](int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int t, i, j;

  /* Hoist the constant weight and create restrict-qualified aliases.
   *
   *  - 'w' is invariant and will typically live in a register.
   *  - A_ and B_ tell the compiler that A and B do not alias and that
   *    each row is laid out contiguously (row-major).
   *  - Using these aliases improves vectorization and allows the
   *    compiler to keep row pointers in registers.
   */
  const DATA_TYPE w = SCALAR_VAL(0.2);
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict B_)[n] = B;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      /* ----------------------------------------------------------
       * Sweep 1: B = Jacobi(A)
       *
       * Tiled over the interior domain [1, _PB_N-1) x [1, _PB_N-1)
       * to improve cache locality. Within each tile we iterate with
       * i outer, j inner so that:
       *   - j traverses contiguous elements (unit stride),
       *   - neighboring rows A[i-1], A[i], A[i+1] stay hot in cache.
       *
       * The tiling does NOT change which points are updated:
       * we still update every (i, j) with 1 <= i,j <= _PB_N-2.
       * ---------------------------------------------------------- */
      for (int ii = 1; ii < _PB_N - 1; ii += JACOBI_2D_TILE_I)
      {
        int i_max = ii + JACOBI_2D_TILE_I;
        if (i_max > _PB_N - 1)
          i_max = _PB_N - 1;

        for (int jj = 1; jj < _PB_N - 1; jj += JACOBI_2D_TILE_J)
        {
          int j_max = jj + JACOBI_2D_TILE_J;
          if (j_max > _PB_N - 1)
            j_max = _PB_N - 1;

          for (i = ii; i < i_max; i++)
          {
            /* Row pointers for the current tile row.
             * Keeping these in registers reduces address arithmetic. */
            DATA_TYPE *restrict Bi    = &B_[i][0];
            DATA_TYPE *restrict Ai    = &A_[i][0];
            DATA_TYPE *restrict Ai_m1 = &A_[i-1][0];
            DATA_TYPE *restrict Ai_p1 = &A_[i+1][0];

            for (j = jj; j < j_max; j++)
            {
              Bi[j] = w * (Ai[j]     +
                           Ai[j-1]   +
                           Ai[j+1]   +
                           Ai_m1[j]  +
                           Ai_p1[j]);
            }
          }
        }
      }

      /* ----------------------------------------------------------
       * Sweep 2: A = Jacobi(B)
       *
       * Same tiling and loop structure as Sweep 1, but reading B
       * and writing A. Sweep 2 starts only after ALL of B has been
       * computed in Sweep 1, preserving the original dependency:
       *   "finish all B updates at time t before any A update at t".
       * ---------------------------------------------------------- */
      for (int ii = 1; ii < _PB_N - 1; ii += JACOBI_2D_TILE_I)
      {
        int i_max = ii + JACOBI_2D_TILE_I;
        if (i_max > _PB_N - 1)
          i_max = _PB_N - 1;

        for (int jj = 1; jj < _PB_N - 1; jj += JACOBI_2D_TILE_J)
        {
          int j_max = jj + JACOBI_2D_TILE_J;
          if (j_max > _PB_N - 1)
            j_max = _PB_N - 1;

          for (i = ii; i < i_max; i++)
          {
            DATA_TYPE *restrict Ai    = &A_[i][0];
            DATA_TYPE *restrict Bi    = &B_[i][0];
            DATA_TYPE *restrict Bi_m1 = &B_[i-1][0];
            DATA_TYPE *restrict Bi_p1 = &B_[i+1][0];

            for (j = jj; j < j_max; j++)
            {
              Ai[j] = w * (Bi[j]     +
                           Bi[j-1]   +
                           Bi[j+1]   +
                           Bi_m1[j]  +
                           Bi_p1[j]);
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