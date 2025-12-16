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
 * These control the spatial tile sizes in the i- and j-dimensions.
 * They can be overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DJACOBI_BLOCK_I=64 -DJACOBI_BLOCK_J=64 ...
 *
 * The defaults are conservative and should work well on most machines.
 * ------------------------------------------------------------------*/
#ifndef JACOBI_BLOCK_I
# define JACOBI_BLOCK_I 32
#endif

#ifndef JACOBI_BLOCK_J
# define JACOBI_BLOCK_J 32
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
   - OpenMP parallelization over space dimensions (if compiled with -fopenmp).
   - 2D blocking in (i,j) using JACOBI_BLOCK_I/J to improve cache locality.
   - Use of restrict-qualified local pointers for better alias analysis.
   - Hoisting of constant SCALAR_VAL(0.2) out of the loops.
   - Row-pointer temporaries inside the innermost loops to reduce address
     arithmetic.
   - Optional OpenMP 'simd' on the innermost loops to encourage vectorization.
*/
static
void kernel_jacobi_2d[[gnu::flatten, gnu::noinline]](int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  /* PolyBench provides possibly adjusted loop bounds via these macros. */
  const int tstepsPB = _PB_TSTEPS;
  const int nPB      = _PB_N;

  /* Constant stencil weight, hoisted out of the loops. */
  const DATA_TYPE w = SCALAR_VAL(0.2);

  /* Local restrict-qualified views of the 2D arrays.
     The 'restrict' qualifier allows the compiler to assume that A_ and B_
     do not alias, which enables more aggressive reordering and vectorization. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict B_)[n] = B;

#pragma scop
  /* Parallel region is created once; all time steps are executed inside.
     If compiled without OpenMP, this pragma is ignored by the compiler,
     and the code naturally runs sequentially. */
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    /* Each thread has its own private 't'. The OpenMP 'for' directives
       inside the loop synchronize threads at the end of each sweep, so
       all threads stay in lock-step over t. */
    for (int t = 0; t < tstepsPB; ++t)
    {
      /* ------------------------------------------------------------------
       * Sweep 1: compute B from A.
       * We iterate over (ii, jj) tiles to improve data locality, and
       * over (i, j) inside each tile. The 'collapse(2)' distributes the
       * tiles across threads.
       * Each (i,j) in the interior 1..nPB-2 is updated exactly once.
       * ----------------------------------------------------------------*/
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
      for (int ii = 1; ii < nPB - 1; ii += JACOBI_BLOCK_I)
        for (int jj = 1; jj < nPB - 1; jj += JACOBI_BLOCK_J)
        {
          const int i_end = (ii + JACOBI_BLOCK_I < nPB - 1) ? (ii + JACOBI_BLOCK_I) : (nPB - 1);
          const int j_end = (jj + JACOBI_BLOCK_J < nPB - 1) ? (jj + JACOBI_BLOCK_J) : (nPB - 1);

          for (int i = ii; i < i_end; ++i)
          {
            /* Pointers to the three rows of A and the corresponding row of B
               used in this i-iteration. This reduces repeated row-base
               address computation in the inner j-loop. */
            DATA_TYPE *restrict Ai   = &A_[i  ][0];
            DATA_TYPE *restrict Aim1 = &A_[i-1][0];
            DATA_TYPE *restrict Aip1 = &A_[i+1][0];
            DATA_TYPE *restrict Bi   = &B_[i  ][0];

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int j = jj; j < j_end; ++j)
            {
              Bi[j] = w * (Ai[j]     +
                           Ai[j - 1] +
                           Ai[j + 1] +
                           Aim1[j]   +
                           Aip1[j]);
            }
          }
        }

      /* ------------------------------------------------------------------
       * Sweep 2: compute A from B.
       * Same blocking and parallelization pattern as above, but with
       * the roles of A and B swapped.
       * ----------------------------------------------------------------*/
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
      for (int ii = 1; ii < nPB - 1; ii += JACOBI_BLOCK_I)
        for (int jj = 1; jj < nPB - 1; jj += JACOBI_BLOCK_J)
        {
          const int i_end = (ii + JACOBI_BLOCK_I < nPB - 1) ? (ii + JACOBI_BLOCK_I) : (nPB - 1);
          const int j_end = (jj + JACOBI_BLOCK_J < nPB - 1) ? (jj + JACOBI_BLOCK_J) : (nPB - 1);

          for (int i = ii; i < i_end; ++i)
          {
            DATA_TYPE *restrict Bi   = &B_[i  ][0];
            DATA_TYPE *restrict Bim1 = &B_[i-1][0];
            DATA_TYPE *restrict Bip1 = &B_[i+1][0];
            DATA_TYPE *restrict Ai   = &A_[i  ][0];

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int j = jj; j < j_end; ++j)
            {
              Ai[j] = w * (Bi[j]     +
                           Bi[j - 1] +
                           Bi[j + 1] +
                           Bim1[j]   +
                           Bip1[j]);
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