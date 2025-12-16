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

   Optimizations applied:
   - Use local restrict-qualified pointers derived from the PolyBench
     array parameters to help the compiler with alias analysis.
   - Use __builtin_assume_aligned to communicate the known alignment
     (PolyBench allocates aligned memory), improving vectorization.
   - Rewrite the inner dot product loop to stream linearly through `r`
     and `y` (one forward, one backward) with simple pointer arithmetic.
   - Remove the temporary array `z[]` and perform the y[0..k-1] update
     in-place by updating symmetric index pairs (i, k-1-i). This keeps
     the same mathematical transformation but halves the memory traffic
     and improves cache locality.
 */
static
void kernel_durbin[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  /* Create local pointers with better aliasing and alignment information.
     Using &r[0] / &y[0] works both when the parameters are true arrays
     or pointers. The alignment assumption relies on PolyBench's
     polybench_alloc_data, which returns at least cache-line aligned data. */
  DATA_TYPE * __restrict r_ptr_base =
      (DATA_TYPE * __restrict)__builtin_assume_aligned(&r[0], 64);
  DATA_TYPE * __restrict y_ptr_base =
      (DATA_TYPE * __restrict)__builtin_assume_aligned(&y[0], 64);

  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  /* Initial conditions. */
  y_ptr_base[0] = -r_ptr_base[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r_ptr_base[0];

  /* Main Durbin recursion. */
  for (k = 1; k < _PB_N; k++) {
    /* beta_k = (1 - alpha_{k-1}^2) * beta_{k-1} */
    beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

    /* Compute:
       sum = sum_{i=0}^{k-1} r[k-1-i] * y[i]

       This is mathematically equivalent to:
       sum = sum_{i=0}^{k-1} r[i] * y[k-1-i]

       We use the latter form because:
       - r[0..k-1] is accessed in strictly increasing order.
       - y[k-1..0] is accessed in strictly decreasing order.
       Both are unit-stride traversals, which are friendlier to caches
       and vectorization than the original mixed indexing. */
    sum = SCALAR_VAL(0.0);
    {
      const DATA_TYPE * __restrict r_it   = r_ptr_base;       /* r[0]     */
      const DATA_TYPE * __restrict r_end  = r_ptr_base + k;   /* r[k]     */
      const DATA_TYPE * __restrict y_it   = y_ptr_base + (k - 1); /* y[k-1] */

      for (; r_it < r_end; ++r_it, --y_it) {
        sum += (*r_it) * (*y_it);
      }
    }

    /* alpha_k = - (r[k] + sum) / beta_k */
    alpha = - (r_ptr_base[k] + sum) / beta;

    /* Original code used a temporary array z[] to hold the updated
       values:
         z[i] = y[i] + alpha * y[k-1-i];
         y[i] = z[i];

       That is, all y[0..k-1] are updated simultaneously from the
       previous y values.

       We can perform this update in-place without an auxiliary array
       by working on symmetric index pairs (i, j), where j = k-1-i.
       For each pair:
         new_y[i] = old_y[i] + alpha * old_y[j];
         new_y[j] = old_y[j] + alpha * old_y[i];
       Both depend only on the old values, so we save them into
       temporaries before writing back.

       When k is odd, there is one middle element i == j; in that case
       the update reduces to:
         new_y[i] = old_y[i] + alpha * old_y[i] = (1 + alpha) * old_y[i].
    */
    {
      int half = k >> 1; /* floor(k/2) */

      /* Update symmetric pairs. No cross-iteration dependencies:
         each iteration touches a disjoint pair (i, k-1-i),
         so this loop is vectorizable. */
      for (i = 0; i < half; ++i) {
        int j = k - 1 - i;
        DATA_TYPE yi = y_ptr_base[i];
        DATA_TYPE yj = y_ptr_base[j];

        y_ptr_base[i] = yi + alpha * yj;
        y_ptr_base[j] = yj + alpha * yi;
      }

      /* Handle the middle element when k is odd. */
      if (k & 1) {
        int mid = half;
        DATA_TYPE ym = y_ptr_base[mid];
        y_ptr_base[mid] = ym + alpha * ym;
      }
    }

    /* Finally set y[k] = alpha_k. */
    y_ptr_base[k] = alpha;
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