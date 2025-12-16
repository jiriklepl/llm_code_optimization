/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

/* Tunable unroll factor for the inner j-loop in the kernel.
 * Can be overridden at compile time via:
 *   -DTRISOLV_J_UNROLL=<positive integer>
 *
 * Values > 1 enable a small blocked + unrolled inner loop.
 * This affects performance only, not correctness.
 */
#ifndef TRISOLV_J_UNROLL
# define TRISOLV_J_UNROLL 4
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* Give the compiler aliasing information without changing behavior. */
  DATA_TYPE (* __restrict__ L_)[n] = L;
  DATA_TYPE * __restrict__ x_ = x;
  DATA_TYPE * __restrict__ b_ = b;

  for (i = 0; i < n; i++)
    {
      x_[i] = -999;
      b_[i] = i;
      for (j = 0; j <= i; j++)
        /* Keep the original integer arithmetic exactly as written to
         * preserve initialization semantics. */
	L_[i][j] = (DATA_TYPE) (i + n - j + 1) * 2 / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv [[gnu::flatten, gnu::noinline]] (int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  /* Local restrict-qualified views to help the optimizer:
   * L_:  pointer to n-element rows
   * x_, b_:  1D vectors
   * This does not change observable behavior; the actual data are the same. */
  DATA_TYPE (* __restrict__ L_)[n] = L;
  DATA_TYPE * __restrict__ x_ = x;
  DATA_TYPE * __restrict__ b_ = b;

  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* Keep the running value of x[i] in a register (scalar accumulator).
       * This preserves the original evaluation order:
       *   xi_0 = b[i]
       *   xi_{k+1} = xi_k - L[i][k] * x[k]
       */
      DATA_TYPE xi = b_[i];

      /* Cache the current row pointer for better locality and less
       * address computation in the inner loop. */
      DATA_TYPE * __restrict__ Li = L_[i];

#if TRISOLV_J_UNROLL > 1
      /* Block + (potentially) unrolled inner loop over j.
       * The order of operations on xi and the sequence of (i,j) pairs
       * is identical to the original code. */
      int limit = i - (i % TRISOLV_J_UNROLL);
      for (j = 0; j < limit; j += TRISOLV_J_UNROLL)
        {
#pragma GCC unroll TRISOLV_J_UNROLL
          for (int k = 0; k < TRISOLV_J_UNROLL; ++k)
            {
              xi -= Li[j + k] * x_[j + k];
            }
        }
#else
      /* Fallback: no blocking/unrolling. */
      j = 0;
#endif

      /* Clean up any remaining iterations (at most TRISOLV_J_UNROLL-1). */
      for (; j < i; j++)
        {
          xi -= Li[j] * x_[j];
        }

      /* Final division, exactly as in the original kernel. */
      x_[i] = xi / Li[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}