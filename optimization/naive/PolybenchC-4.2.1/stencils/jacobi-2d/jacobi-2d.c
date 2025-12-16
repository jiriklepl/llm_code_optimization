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


/* Array initialization.
 *
 * Optimization:
 *  - Use local restrict-qualified row pointers to help the compiler
 *    with alias analysis and enable better vectorization.
 *  - Keep the arithmetic exactly as in the original code to preserve
 *    floating‑point semantics.
 */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  /* POLYBENCH_2D guarantees A and B are laid out as contiguous
     row-major n x n arrays. Create row pointers for faster access. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict B_)[n] = B;

  for (i = 0; i < n; i++)
    {
      DATA_TYPE *restrict Ai = A_[i];
      DATA_TYPE *restrict Bi = B_[i];

#pragma GCC ivdep
#ifdef _OPENMP
# pragma omp simd
#endif
      for (j = 0; j < n; j++)
      {
        /* Same arithmetic as original code (no strength reduction
           such as replacing division by multiplication). */
	Ai[j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	Bi[j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
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
 *
 * Optimizations applied:
 *  - Use local restrict-qualified row pointers for A and B to
 *    improve alias analysis and enable more aggressive vectorization.
 *  - Hoist the constant scalar 0.2 out of the loops.
 *  - Add OpenMP parallelization over the outer spatial loop (i)
 *    when compiled with -fopenmp. Each (i,j) update is independent,
 *    so this preserves semantics.
 *  - Add SIMD/vectorization hints on the inner loop (j).
 *  - Preserve original loop bounds and update formulas exactly.
 */
static
void kernel_jacobi_2d [[gnu::flatten, gnu::noinline]] (int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  /* Use the PolyBench loop-bound macros to preserve original behavior. */
  const int tsteps_ = _PB_TSTEPS;
  const int n_pb    = _PB_N;

  /* A and B are n x n VLAs; build local restrict-qualified row pointers.
     Using the parameter 'n' for the second dimension keeps the type
     consistent with the function signature. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict B_)[n] = B;

  /* Hoist constant multiplier. */
  const DATA_TYPE c = SCALAR_VAL(0.2);

  int t, i, j;

#pragma scop
  for (t = 0; t < tsteps_; t++)
    {
      /* Sweep 1: update B from A.
         Each (i,j) update depends only on A (previous time step),
         so the loop nest is fully parallelizable and vectorizable. */

#ifdef _OPENMP
# pragma omp parallel for private(j) schedule(static) if (n_pb >= 32)
#endif
      for (i = 1; i < n_pb - 1; i++)
        {
          /* Row pointers for this and neighboring rows. */
          DATA_TYPE *restrict Bi   = B_[i];
          DATA_TYPE *restrict Ai   = A_[i];
          DATA_TYPE *restrict Aim1 = A_[i-1];
          DATA_TYPE *restrict Aip1 = A_[i+1];

          /* Tell the compiler there are no loop-carried dependencies
             along j, enabling SIMD vectorization. */
#pragma GCC ivdep
#ifdef _OPENMP
# pragma omp simd
#endif
          for (j = 1; j < n_pb - 1; j++)
            {
              /* Same stencil as the original:
                 B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1]
                                            + A[i+1][j] + A[i-1][j]); */
              Bi[j] = c * (Ai[j]     +
                           Ai[j-1]   +
                           Ai[j+1]   +
                           Aip1[j]   +
                           Aim1[j]);
            }
        }

      /* Sweep 2: update A from B.
         After the first sweep, B holds the complete new time step, so
         this sweep is likewise fully parallelizable. */

#ifdef _OPENMP
# pragma omp parallel for private(j) schedule(static) if (n_pb >= 32)
#endif
      for (i = 1; i < n_pb - 1; i++)
        {
          DATA_TYPE *restrict Ai   = A_[i];
          DATA_TYPE *restrict Bi   = B_[i];
          DATA_TYPE *restrict Bim1 = B_[i-1];
          DATA_TYPE *restrict Bip1 = B_[i+1];

#pragma GCC ivdep
#ifdef _OPENMP
# pragma omp simd
#endif
          for (j = 1; j < n_pb - 1; j++)
            {
              /* Same stencil as the original:
                 A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1]
                                            + B[i+1][j] + B[i-1][j]); */
              Ai[j] = c * (Bi[j]     +
                           Bi[j-1]   +
                           Bi[j+1]   +
                           Bip1[j]   +
                           Bim1[j]);
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