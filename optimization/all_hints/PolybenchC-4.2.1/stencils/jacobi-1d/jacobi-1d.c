/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h> /* Added for optional OpenMP parallelization */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"

/* Tuneable OpenMP chunk size for spatial loops.
 * Adjust at compile time (e.g. with -DJACOBI_1D_OMP_CHUNK_SIZE=...) to
 * better match a specific machine's cache and core configuration. */
#ifndef JACOBI_1D_OMP_CHUNK_SIZE
#  define JACOBI_1D_OMP_CHUNK_SIZE 1024
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
    {
      A[i] = ((DATA_TYPE) i+ 2) / n;
      B[i] = ((DATA_TYPE) i+ 3) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d [[gnu::flatten, gnu::noinline]] (int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  /* The parameters tsteps and n are intentionally unused directly:
   * PolyBench provides sanitized loop bounds via the _PB_* macros that
   * may clamp these values. Suppress potential warnings. */
  (void)tsteps;
  (void)n;

#pragma scop
  /* Constant factor used in the stencil. Using a single local constant
   * avoids re-materializing the literal and documents the intent. */
  const DATA_TYPE c = (DATA_TYPE)0.33333;

  /* Local restrict-qualified pointers give the compiler stronger
   * aliasing guarantees than the generic PolyBench array types
   * and improve auto-vectorization. A and B truly do not alias in
   * this benchmark (they are separate arrays in main). */
  DATA_TYPE * restrict A_local = A;
  DATA_TYPE * restrict B_local = B;

  const int N_inner = _PB_N;
  const int T_inner = _PB_TSTEPS;

  /* Parallelize over the spatial dimension for each time step.
   *
   * The temporal loop (t) must remain logically sequential because
   * each step depends on the results of the previous one. Inside a
   * given time step, iterations over i are independent, so they are
   * distributed across threads using 'omp for'.
   *
   * If the program is compiled without -fopenmp, all OpenMP pragmas
   * are ignored by the compiler, and this region executes as a normal
   * sequential block, preserving the original semantics. */
#pragma omp parallel
  {
    for (int t = 0; t < T_inner; ++t)
    {
      /* First Jacobi sweep: update B from the current A. Every i in
       * [1, N_inner-2] is independent, so we can safely parallelize.
       *
       * The implicit barrier at the end of 'omp for' ensures all
       * updates to B are visible before the next loop starts, which
       * preserves the original sequential ordering between the two
       * sweeps for each time step. */
#pragma omp for schedule(static, JACOBI_1D_OMP_CHUNK_SIZE)
      for (int i = 1; i < N_inner - 1; ++i)
      {
        /* Load into temporaries to reduce redundant memory accesses
         * and to present a simple dependency pattern to the vectorizer. */
        const DATA_TYPE a_im1 = A_local[i - 1];
        const DATA_TYPE a_i   = A_local[i];
        const DATA_TYPE a_ip1 = A_local[i + 1];

        B_local[i] = (a_im1 + a_i + a_ip1) * c;
      }

      /* Second Jacobi sweep: update A from the new B.
       *
       * Again, all iterations in i are independent. The implicit
       * barrier at the end of this 'omp for' also ensures that all
       * threads finish time step t before any of them start step t+1,
       * since they must all reach the end of the loop body. */
#pragma omp for schedule(static, JACOBI_1D_OMP_CHUNK_SIZE)
      for (int i = 1; i < N_inner - 1; ++i)
      {
        const DATA_TYPE b_im1 = B_local[i - 1];
        const DATA_TYPE b_i   = B_local[i];
        const DATA_TYPE b_ip1 = B_local[i + 1];

        A_local[i] = (b_im1 + b_i + b_ip1) * c;
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
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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