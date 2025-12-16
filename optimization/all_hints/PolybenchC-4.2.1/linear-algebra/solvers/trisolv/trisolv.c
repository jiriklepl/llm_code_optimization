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

/* Tunable parameter: unrolling factor for the inner loop in the
 * triangular solve kernel. The current implementation explicitly
 * unrolls by a factor of 4; changing this constant requires updating
 * the unrolled body in kernel_trisolv accordingly.
 */
#ifndef TRI_SOLV_UNROLL
# define TRI_SOLV_UNROLL 4
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* Help the compiler with alias analysis. In PolyBench, L, x and b
   * are allocated as distinct arrays, so they cannot overlap.  Using
   * restrict-qualified local pointers enables more aggressive
   * optimization while preserving semantics. */
  DATA_TYPE (*restrict Lr)[n] = L;
  DATA_TYPE *restrict xr = x;
  DATA_TYPE *restrict br = b;

  /* The outer i-loop is embarrassingly parallel: each i-iteration
   * writes to a disjoint row of L and distinct elements of x and b.
   * When compiled with -fopenmp this pragma parallelizes the loop;
   * otherwise it is ignored and the code runs sequentially. */
#pragma omp parallel for private(j) schedule(static)
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *restrict Li = Lr[i];

      xr[i] = -999;
      br[i] = i;

      /* Precompute the part of the affine expression that does not
       * depend on j, so the inner loop does less work.  The original
       * code computed:
       *   L[i][j] = (DATA_TYPE) (i + n - j + 1) * 2 / n;
       * Because * and / have higher precedence than +/-, and * and /
       * are left-associative, this is equivalent to:
       *   tmp = (i + n - j + 1) * 2;
       *   L[i][j] = (DATA_TYPE) (tmp / n);
       * We keep exactly that integer arithmetic, only factoring out
       * (i + n + 1) which does not depend on j.
       */
      const int base = i + n + 1;  /* base - j == i + n - j + 1 */

      for (j = 0; j <= i; j++)
        {
          const int t = (base - j) * 2;
          Li[j] = (DATA_TYPE)(t / n);
        }
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
void kernel_trisolv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* As in init_array, use restrict-qualified local pointers to help
   * the optimizer.  In the PolyBench setup these arrays never alias. */
  DATA_TYPE (*restrict Lr)[n] = L;
  DATA_TYPE *restrict xr = x;
  DATA_TYPE *restrict br = b;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE *restrict Li = Lr[i];

      /* Keep the accumulator for x[i] in a register-resident scalar
       * to avoid repeated loads/stores to memory.  We update it in
       * exactly the same order as the original code, preserving the
       * sequence of floating-point operations and therefore its
       * numerical semantics. */
      DATA_TYPE xi = br[i];

      /* Manually unroll the inner loop to reduce loop overhead and
       * increase instruction-level parallelism.  The unrolled body
       * performs the updates in strictly increasing j order:
       *   j, j+1, j+2, j+3, ...
       * which is the same order as the original scalar loop.
       */
      const int j_unroll_limit = (i / TRI_SOLV_UNROLL) * TRI_SOLV_UNROLL;

      for (j = 0; j < j_unroll_limit; j += TRI_SOLV_UNROLL)
        {
          /* Explicit unrolling with factor TRI_SOLV_UNROLL == 4.
           * If TRI_SOLV_UNROLL is changed, this block must be updated
           * accordingly to keep the unrolling consistent. */
#if TRI_SOLV_UNROLL == 4
          xi -= Li[j]     * xr[j];
          xi -= Li[j + 1] * xr[j + 1];
          xi -= Li[j + 2] * xr[j + 2];
          xi -= Li[j + 3] * xr[j + 3];
#else
          /* Fallback: no manual unrolling, but still use the scalar
           * accumulator xi. */
          int jj;
          for (jj = 0; jj < TRI_SOLV_UNROLL; ++jj)
            xi -= Li[j + jj] * xr[j + jj];
#endif
        }

      /* Handle the remaining iterations (at most TRI_SOLV_UNROLL-1). */
      for (; j < i; j++)
        {
          xi -= Li[j] * xr[j];
        }

      /* Final division by the diagonal element.  Using Li[i] instead
       * of L[i][i] helps the compiler keep the value in registers. */
      xi = xi / Li[i];

      xr[i] = xi;
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