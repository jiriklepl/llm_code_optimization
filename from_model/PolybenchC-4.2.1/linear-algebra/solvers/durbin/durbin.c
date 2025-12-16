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

  (void)j; /* j is unused but kept to preserve original structure */

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


/*
 * Main computational kernel. The whole function will be timed,
 * including the call and return.
 *
 * Optimizations compared to the original version:
 *   - Eliminate the temporary array z[N]. The original code did two
 *     passes over y[0..k-1]:
 *         z[i] = y[i] + alpha * y[k-1-i];
 *         y[i] = z[i];
 *     We replace this with an equivalent in-place, symmetric pair update
 *     on (i, k-1-i) using only scalar temporaries. This preserves the
 *     exact per-element arithmetic while halving the number of passes
 *     over y and removing all accesses to z[.].
 *
 *   - Use local __restrict pointers to help the compiler with
 *     alias analysis and auto-vectorization.
 *
 *   - Keep the exact loop bounds and scalar recurrence structure
 *     (alpha, beta, y) so that the numerical behavior is preserved.
 */
static
void kernel_durbin[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  /* Local pointer aliases marked restrict to help the optimizer.
     POLYBENCH_1D parameters decay to pointers, so this is safe. */
  DATA_TYPE * __restrict r_ = r;
  DATA_TYPE * __restrict y_ = y;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  /* Initialization (k = 0 step) – unchanged semantics. */
  y_[0] = -r_[0];
  beta  = SCALAR_VAL(1.0);
  alpha = -r_[0];

  /* Main Durbin recursion: k must remain sequential. */
  for (k = 1; k < _PB_N; k++)
  {
    /* beta = (1 - alpha^2) * beta; */
    beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

    /* sum = sum_{i=0}^{k-1} r[k-1-i] * y[i]; */
    sum = SCALAR_VAL(0.0);
    for (i = 0; i < k; i++)
    {
      /* Unit-stride in y_, reverse unit-stride in r_. */
      sum += r_[k - i - 1] * y_[i];
    }

    /* alpha_k = -(r[k] + sum) / beta_k; */
    alpha = -(r_[k] + sum) / beta;

    /*
     * In-place symmetric update of y_[0..k-1].
     *
     * Original code:
     *   for (i = 0; i < k; i++)
     *     z[i] = y[i] + alpha * y[k-i-1];
     *   for (i = 0; i < k; i++)
     *     y[i] = z[i];
     *
     * For each i, the final value is:
     *   y_new[i] = y_old[i] + alpha * y_old[k-1-i]
     *
     * The pair (i, j = k-1-i) is symmetric:
     *   y_new[i] = y_old[i] + alpha * y_old[j]
     *   y_new[j] = y_old[j] + alpha * y_old[i]
     *
     * We can compute both y_new[i] and y_new[j] from the old values
     * y_old[i], y_old[j] using scalar temporaries, with no extra array.
     * Each index 0..k-1 participates in exactly one such pair, and
     * different iterations use disjoint indices, so this is
     * free of cross-iteration dependencies.
     */
    {
      int half = k >> 1; /* integer division: k / 2 */

      /* Pairwise updates for (i, k-1-i). */
      for (i = 0; i < half; i++)
      {
        int j = k - 1 - i;
        DATA_TYPE yi = y_[i];
        DATA_TYPE yj = y_[j];

        /* These two assignments reproduce:
           z[i] = yi + alpha * yj;
           z[j] = yj + alpha * yi;
           y[i] = z[i]; y[j] = z[j]; */
        y_[i] = yi + alpha * yj;
        y_[j] = yj + alpha * yi;
      }

      /* For odd k, there is a center index c = k/2 such that
         c == k-1-c. In the original code:
             z[c] = y[c] + alpha * y[c];
             y[c] = z[c];
         which is y[c] = y[c] + alpha * y[c]. */
      if (k & 1)
      {
        int c = half; /* same as k / 2 for odd k */
        DATA_TYPE yc = y_[c];
        y_[c] = yc + alpha * yc;
      }
    }

    /* Append the current alpha at y[k] – unchanged. */
    y_[k] = alpha;
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