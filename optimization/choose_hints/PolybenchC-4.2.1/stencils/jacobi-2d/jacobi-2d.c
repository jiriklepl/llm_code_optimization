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
   - Use restrict-qualified local aliases (A_ and B_) to help the
     compiler with alias analysis and vectorization.
   - Cache row pointers (Ai, Bi, etc.) inside the i-loop to reduce
     address arithmetic and improve cache locality.
   - Optional OpenMP parallelization over the spatial loops, using a
     single parallel region around the time loop to reduce thread
     creation overhead (enabled when compiled with -fopenmp).
   - Use GCC's ivdep pragma on the innermost loop to encourage
     vectorization when using GCC/Clang.

   The numerical scheme and update order are preserved exactly:
   for each time step, we first compute B from A over the full interior
   domain, then compute A from B over the same domain.
*/
static
void kernel_jacobi_2d[[gnu::flatten, gnu::noinline]](int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  /* Local restrict-qualified aliases for better optimization.
     A_ and B_ describe the same memory as A and B, but tell the
     compiler that they do not alias each other.  The leading
     dimension 'n' is passed explicitly, so we can use it in the
     VLA type. */
  DATA_TYPE (* __restrict A_)[n] = A;
  DATA_TYPE (* __restrict B_)[n] = B;

  /* Constant stencil weight. */
  const DATA_TYPE c0 = SCALAR_VAL(0.2);

  int t, i, j;

#pragma scop
#if defined(_OPENMP)
  /* Single parallel region to amortize thread start-up costs.
     Each iteration of the time loop performs two sweeps:
       1) A -> B
       2) B -> A
     The implicit barrier at the end of each 'omp for' ensures
     correct ordering between the sweeps and between time steps. */
#pragma omp parallel private(t,i,j)
  {
#endif

  for (t = 0; t < _PB_TSTEPS; t++)
    {
      /* First sweep: update B based on the current values of A. */
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (i = 1; i < _PB_N - 1; i++)
	{
	  /* Cache frequently used row pointers to improve locality and
	     reduce address arithmetic inside the inner loop. */
	  DATA_TYPE *Bi   = B_[i];
	  DATA_TYPE *Ai   = A_[i];
	  DATA_TYPE *Aim1 = A_[i-1];
	  DATA_TYPE *Aip1 = A_[i+1];

#if defined(__GNUC__)
#pragma GCC ivdep
#endif
	  for (j = 1; j < _PB_N - 1; j++)
	    Bi[j] = c0 * (Ai[j]     +
			  Ai[j-1]   +
			  Ai[j+1]   +
			  Aim1[j]   +
			  Aip1[j]);
	}

      /* Second sweep: update A based on the new values of B. */
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (i = 1; i < _PB_N - 1; i++)
	{
	  DATA_TYPE *Ai   = A_[i];
	  DATA_TYPE *Bi   = B_[i];
	  DATA_TYPE *Bim1 = B_[i-1];
	  DATA_TYPE *Bip1 = B_[i+1];

#if defined(__GNUC__)
#pragma GCC ivdep
#endif
	  for (j = 1; j < _PB_N - 1; j++)
	    Ai[j] = c0 * (Bi[j]     +
			  Bi[j-1]   +
			  Bi[j+1]   +
			  Bim1[j]   +
			  Bip1[j]);
	}
    }

#if defined(_OPENMP)
  } /* end parallel */
#endif
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