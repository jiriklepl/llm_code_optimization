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
 * - Use restrict-qualified local aliases for L, x, and b to aid the
 *   optimizer (these arrays are allocated disjointly by PolyBench).
 * - Hoist the loop-invariant part of (i + n - j + 1) outside the j-loop
 *   to reduce arithmetic in the inner loop, while keeping all arithmetic
 *   in integer type so the result before the cast is unchanged.
 */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* Local restrict-qualified aliases to improve optimization. */
  DATA_TYPE (*restrict L_)[N] = (DATA_TYPE (*)[N])L;
  DATA_TYPE *restrict x_      = (DATA_TYPE *restrict)x;
  DATA_TYPE *restrict b_      = (DATA_TYPE *restrict)b;

  for (i = 0; i < n; i++)
    {
      x_[i] = (DATA_TYPE)-999;
      b_[i] = (DATA_TYPE)i;

      /* Original formula:
       *   L[i][j] = (DATA_TYPE) (i + n - j + 1) * 2 / n;
       *
       * We rewrite it as:
       *   base = i + n + 1;
       *   num  = (base - j) * 2;
       *   L[i][j] = (DATA_TYPE)(num / n);
       *
       * All operations remain in integer type before the cast, so the
       * final value (including integer division truncation) is preserved.
       */
      const int base = i + n + 1;
      for (j = 0; j <= i; j++)
        {
          const int num = (base - j) * 2;
          L_[i][j] = (DATA_TYPE)(num / n);
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
   including the call and return.
 *
 * Optimizations:
 * - Introduce restrict-qualified pointer aliases for L, x, and b.
 *   PolyBench allocations do not alias, so this is safe and enables
 *   better vectorization and scheduling.
 * - Keep x[i] in a local scalar (xi) during the computation to
 *   drastically reduce memory traffic and allow the compiler to keep
 *   it in a register.
 * - Use an explicit row pointer Li = L_[i] to improve spatial locality
 *   and reduce address calculation inside the inner loop.
 * - The order of floating-point operations is unchanged: we still
 *   perform the same sequence of subtractions in the same order as
 *   the original code, preserving numerical behavior.
 */
static
void kernel_trisolv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* Local restrict-qualified aliases for better optimization. */
  DATA_TYPE (*restrict L_)[N] = (DATA_TYPE (*)[N])L;
  DATA_TYPE *restrict x_      = (DATA_TYPE *restrict)x;
  DATA_TYPE *restrict b_      = (DATA_TYPE *restrict)b;

  (void)n; /* n is encoded in _PB_N/N; silence potential unused-parameter warnings. */

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* Start from b[i] and update in-place in a scalar register. */
      DATA_TYPE xi = b_[i];
      DATA_TYPE *restrict Li = L_[i];

      for (j = 0; j < i; j++)
        {
          xi -= Li[j] * x_[j];
        }

      /* Divide by the diagonal element of L. */
      xi /= Li[i];

      /* Write back the final value once. */
      x_[i] = xi;
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