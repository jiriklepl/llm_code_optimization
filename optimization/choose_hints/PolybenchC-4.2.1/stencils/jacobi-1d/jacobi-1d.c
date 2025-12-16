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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
    {
      A[i] = ((DATA_TYPE) i + 2) / n;
      B[i] = ((DATA_TYPE) i + 3) / n;
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
   including the call and return.

   Optimizations:
   - Use local restrict-qualified pointers to help the compiler
     assume A and B do not alias and better optimize memory access.
   - Hoist the stencil coefficient and inner loop bound out of the
     inner loops.
   - Optionally parallelize the spatial loops with OpenMP. If the code
     is compiled without -fopenmp, the OpenMP pragmas are ignored and
     the sequential semantics are preserved.
*/
static
void kernel_jacobi_1d[[gnu::flatten, gnu::noinline]](int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

  /* Local restrict pointers: A and B are distinct PolyBench arrays and
     never alias, so this is safe and allows better vectorization. */
  DATA_TYPE *restrict a = A;
  DATA_TYPE *restrict b = B;

  /* Hoist loop invariants. */
  const DATA_TYPE c = (DATA_TYPE)0.33333;
  const int n_inner = _PB_N - 1;

#pragma scop
  /* Single parallel region to amortize OpenMP overhead.
     Time dimension (t) remains sequential; space dimension (i) is
     distributed across threads. Implicit barriers at the end of each
     'omp for' ensure correct ordering between sweeps and between
     time steps. */
#pragma omp parallel private(t, i) shared(a, b)
  {
    for (t = 0; t < _PB_TSTEPS; t++)
      {
        /* First sweep: read from a (old time level), write to b. */
#pragma omp for schedule(static)
        for (i = 1; i < n_inner; i++)
          b[i] = c * (a[i-1] + a[i] + a[i+1]);

        /* Second sweep: read from b, write back to a. */
#pragma omp for schedule(static)
        for (i = 1; i < n_inner; i++)
          a[i] = c * (b[i-1] + b[i] + b[i+1]);
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