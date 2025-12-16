/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations:
 *  - Remove the temporary array z[] by updating y[] in-place using
 *    symmetric index pairs (i, k-1-i). This preserves the original
 *    values needed for the update while eliminating an O(N^2) copy.
 *  - Rewrite the inner sum loop with a simple decrementing index on r[]
 *    to improve memory access patterns and make the loop easier to
 *    optimize/autovectorize.
 */
static
void kernel_durbin[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

  for (k = 1; k < _PB_N; k++) {
    /* Same recurrence as original: beta = (1 - alpha^2) * beta */
    beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

    /* Compute:
       sum = \sum_{i=0}^{k-1} r[k-1-i] * y[i]
       Use a separate running index for r to simplify addressing. */
    sum = SCALAR_VAL(0.0);
    {
      int rk = k - 1;
      for (i = 0; i < k; i++, rk--) {
        sum += r[rk] * y[i];
      }
    }

    alpha = - (r[k] + sum) / beta;

    /* Original code:
         for (i=0; i<k; i++)
           z[i] = y[i] + alpha * y[k-i-1];
         for (i=0; i<k; i++)
           y[i] = z[i];
       This can be done in-place by updating symmetric pairs (i, j)
       where j = k-1-i, using temporaries to preserve old values:

         new_y[i] = old_y[i] + alpha * old_y[j]
         new_y[j] = old_y[j] + alpha * old_y[i]

       The central element when k is odd (i == j) satisfies:
         new_y[c] = old_y[c] + alpha * old_y[c] = (1+alpha)*old_y[c]
       and is handled separately.

       This transformation is mathematically identical but:
         - removes the extra temporary array z[]
         - removes an additional O(k) copy per iteration
         - improves cache locality and reduces memory traffic.
    */
    {
      int half = k >> 1;       /* floor(k / 2) */
      int k_minus_1 = k - 1;

      /* Update symmetric pairs (i, j) with i < j. */
      for (i = 0; i < half; i++) {
        int j = k_minus_1 - i;
        DATA_TYPE yi = y[i];
        DATA_TYPE yj = y[j];

        y[i] = yi + alpha * yj;
        y[j] = yj + alpha * yi;
      }

      /* If k is odd, handle the middle element (i == k-1-i). */
      if (k & 1) {
        int mid = half;
        y[mid] = y[mid] + alpha * y[mid];
      }
    }

    y[k] = alpha;
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}