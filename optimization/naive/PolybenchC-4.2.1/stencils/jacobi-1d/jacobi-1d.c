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
  /* Parameters are kept for interface compatibility; the PolyBench
     macros _PB_TSTEPS and _PB_N provide the effective problem size. */
  (void)tsteps;
  (void)n;

  /* Use local restrict-qualified pointers to give the compiler
     stronger aliasing information. This allows more aggressive
     vectorization and reordering while preserving semantics. */
  DATA_TYPE * __restrict__ a = A;
  DATA_TYPE * __restrict__ b = B;

  /* Hoist loop-invariant values out of the time loop to avoid
     recomputation. */
  const int tsteps_eff = _PB_TSTEPS;
  const int n_eff      = _PB_N;

  /* We iterate over interior points: i = 1 .. n_eff-2.
     Using an exclusive upper bound (inner_end) makes the loop
     bounds slightly simpler and friendlier for vectorization. */
  const int inner_end  = n_eff - 1; /* loop condition: i < inner_end */

  /* Stencil coefficient, kept in a register for all iterations. */
  const DATA_TYPE c    = (DATA_TYPE)0.33333;

#pragma scop
  for (int t = 0; t < tsteps_eff; ++t)
    {
      /* First sweep: update B from A.
         There are no loop-carried dependencies on B within this loop,
         and A is read-only here, so the iterations are independent. */
#if defined(__GNUC__)
#pragma GCC ivdep          /* assert no loop-carried deps: aid vectorizer */
#pragma GCC unroll 8       /* encourage modest unrolling for ILP */
#endif
      for (int i = 1; i < inner_end; ++i)
        b[i] = c * (a[i-1] + a[i] + a[i+1]);

      /* Second sweep: update A from B.
         Same independence properties as the first sweep. */
#if defined(__GNUC__)
#pragma GCC ivdep
#pragma GCC unroll 8
#endif
      for (int i = 1; i < inner_end; ++i)
        a[i] = c * (b[i-1] + b[i] + b[i+1]);
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