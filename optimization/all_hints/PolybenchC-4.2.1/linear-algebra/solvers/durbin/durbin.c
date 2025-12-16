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
  int i;

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
 * Optimizations vs. the reference version:
 *   - Remove the temporary array z[] and perform the update of y in-place
 *     using symmetric pair updates. This cuts memory traffic roughly in half
 *     and improves cache behavior.
 *   - Use restricted local pointers to help the compiler with alias analysis.
 *   - Use simple, affine index expressions and (optional) SIMD hints to aid
 *     vectorization of the inner loops.
 *
 * The mathematical algorithm is unchanged; only the way we organize
 * the computations and memory accesses is different.
 */
static
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  /* Use local restricted pointers for better aliasing information.
     r is read-only; y is updated in-place. */
  const DATA_TYPE * restrict r_ = r;
  DATA_TYPE       * restrict y_ = y;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int k;

#pragma scop
  /* Initial step of Durbin's recursion. */
  y_[0] = -r_[0];
  beta  = SCALAR_VAL(1.0);
  alpha = -r_[0];

  /* Main recursion loop. Each iteration increases the effective
     system size from k to k+1. */
  for (k = 1; k < _PB_N; k++) {
    /* Update beta. */
    beta *= (SCALAR_VAL(1.0) - alpha * alpha);

    /* Compute
         sum = sum_{i=0..k-1} r[k-i-1] * y[i]
       using two indices so that the compiler sees affine accesses. */
    sum = SCALAR_VAL(0.0);
    {
      int i;
      int j = k - 1;

      /* This is a pure reduction, safe to SIMD-vectorize.
         When compiled without OpenMP support, this pragma is ignored. */
#pragma omp simd reduction(+:sum)
      for (i = 0; i < k; ++i, --j) {
        sum += r_[j] * y_[i]; /* r_[j] == r[k-1-i] */
      }
    }

    /* Update reflection coefficient alpha. */
    alpha = -(r_[k] + sum) / beta;

    /* Original code:
         for (i = 0; i < k; i++)
           z[i] = y[i] + alpha * y[k-i-1];
         for (i = 0; i < k; i++)
           y[i] = z[i];
       This uses a temporary array z[] to avoid overwriting y.
       We can perform the same transformation in-place by updating
       symmetric pairs (i, j = k-1-i) together:

         new_y[i] = y[i] + alpha * y[j];
         new_y[j] = y[j] + alpha * y[i];

       Since each pair uses only the original y[i], y[j], we first
       load them into temporaries and then write both updated values.
       For odd k, the middle element (i == j) is handled separately,
       reproducing y[i] = y[i] + alpha * y[i]. This is mathematically
       equivalent to the reference implementation. */

    int half = k >> 1;     /* floor(k/2) */
    int i_lo = 0;
    int i_hi = k - 1;

    /* Update symmetric pairs (i_lo, i_hi) with i_lo < i_hi. */
#pragma omp simd
    for (; i_lo < half; ++i_lo, --i_hi) {
      DATA_TYPE yi = y_[i_lo];
      DATA_TYPE yj = y_[i_hi];

      y_[i_lo] = yi + alpha * yj;
      y_[i_hi] = yj + alpha * yi;
    }

    /* If k is odd, there is a single middle element with i == k/2.
       For that index, the reference code computes:
         y[mid] = y[mid] + alpha * y[mid];
       We reproduce the same operation order (mul then add). */
    if (k & 1) {
      int mid = half;
      DATA_TYPE ym = y_[mid];
      y_[mid] = ym + alpha * ym;
    }

    /* Append alpha as the new last element. */
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