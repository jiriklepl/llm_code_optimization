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


/* Array initialization.
 *
 * Optimizations:
 * - Add restrict-qualified pointers to help the compiler with alias analysis.
 * - Cache the current row pointer L[i] into a local pointer (Li) to
 *   avoid recomputing the address of L[i][j] inside the inner loop.
 * - Precompute (i + n + 1) as an integer "base" and use (base - j)
 *   instead of (i + n - j + 1). This is algebraically identical at
 *   the integer level and does not change floating‑point evaluation
 *   order of the later *2/n operations.
 */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n) restrict,
		DATA_TYPE POLYBENCH_1D(x,N,n) restrict,
		DATA_TYPE POLYBENCH_1D(b,N,n) restrict)
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = -999;
      b[i] =  i ;

      /* Cache row pointer for better locality and fewer address computations. */
      DATA_TYPE * restrict Li = L[i];
      int base = i + n + 1;

      for (j = 0; j <= i; j++)
        /* Same expression structure as original:
         * (DATA_TYPE)(base - j) * 2 / n == (DATA_TYPE)(i+n-j+1)*2/n
         */
        Li[j] = (DATA_TYPE)(base - j) * 2 / n;
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
   including the call and return.
 *
 * Optimizations:
 * - Add restrict-qualified parameters so the compiler can assume
 *   L, x, and b do not alias. This is true for this benchmark and
 *   greatly helps vectorization and scheduling.
 * - Use a local accumulator 'sum' instead of repeatedly updating x[i]
 *   in memory. This keeps the partial sum in a register, reducing
 *   memory traffic. The update order is preserved, so floating-point
 *   results are bit-identical to the original loop.
 * - Cache the current row pointer L[i] as 'Li' to avoid recomputing
 *   the base address inside the inner loop.
 * - Manually unroll the inner loop by a factor of 4 to improve ILP
 *   and reduce loop overhead. The operations are kept in the same
 *   order as the original j-loop, preserving numerical semantics.
 * - The triangular dependency across i is preserved; only the inner
 *   dot product loop is optimized.
 */
static
void kernel_trisolv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n) restrict,
		    DATA_TYPE POLYBENCH_1D(x,N,n) restrict,
		    DATA_TYPE POLYBENCH_1D(b,N,n) restrict)
{
  int i, j;

  (void)n; /* 'n' is kept for interface compatibility; _PB_N is used. */

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* Pointer to the current row of L for faster access. */
      DATA_TYPE * restrict Li = L[i];

      /* Local accumulator holds the same value that x[i] would hold
       * in the original code, but kept in a register.
       */
      DATA_TYPE sum = b[i];

      /* Manually unrolled inner loop: process 4 elements per iteration.
       * 'limit' is the largest multiple of 4 <= i.
       */
      int limit = i & ~3; /* equivalent to: (i / 4) * 4 */
      j = 0;

      /* There are no loop-carried dependencies on 'j' within this
       * inner loop (only a reduction on 'sum'), and L, x, b do not
       * alias due to 'restrict'. This hint encourages vectorization.
       */
#pragma GCC ivdep
      for (; j < limit; j += 4)
      {
        sum -= Li[j]   * x[j];
        sum -= Li[j+1] * x[j+1];
        sum -= Li[j+2] * x[j+2];
        sum -= Li[j+3] * x[j+3];
      }

      /* Handle the remaining elements (if i is not a multiple of 4). */
      for (; j < i; j++)
      {
        sum -= Li[j] * x[j];
      }

      /* Final division by the diagonal element. */
      sum /= Li[i];
      x[i] = sum;
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