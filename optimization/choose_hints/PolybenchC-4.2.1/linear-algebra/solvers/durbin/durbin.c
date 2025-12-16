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
      r[i] = (n + 1 - i);
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
   including the call and return. */
static
void kernel_durbin[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  /* Removed the large temporary array z[N] to improve cache behavior and
     reduce stack usage. The corresponding computation is now done in-place
     using a symmetric update (see comments below). */

  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  /* Initialization of Durbin's recursion. */
  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

  for (k = 1; k < _PB_N; k++) {

    /* Update beta:
       beta = (1 - alpha^2) * beta.
       Written this way to reuse alpha*alpha and avoid redundant multiplications. */
    const DATA_TYPE alpha_sq = alpha * alpha;
    beta *= (SCALAR_VAL(1.0) - alpha_sq);

    /* Compute:
         sum = sum_{i=0}^{k-1} r[k-i-1] * y[i]
       We restructure the loop to use two pointers that walk contiguous memory
       (one forward in y, one backward in r). This improves locality and helps
       auto-vectorization. */
    sum = SCALAR_VAL(0.0);

    const DATA_TYPE * __restrict r_ptr = r + (k - 1); /* starts at r[k-1] */
    const DATA_TYPE * __restrict y_ptr = y;           /* starts at y[0]   */

    for (i = 0; i < k; i++) {
      sum += (*r_ptr--) * (*y_ptr++);
    }

    /* Update alpha:
         alpha = - (r[k] + sum) / beta
    */
    alpha = - (r[k] + sum) / beta;

    /* Update y[0..k-1].
       Original code:
         for (i=0; i<k; i++)
           z[i] = y[i] + alpha * y[k-i-1];
         for (i=0; i<k; i++)
           y[i] = z[i];

       Mathematically:
         new_y[i] = old_y[i] + alpha * old_y[k-1-i]

       This update couples symmetric indices (i, j = k-1-i). We can perform the
       same transformation in-place without a temporary buffer by updating pairs:

         new_y[i] = old_y[i] + alpha * old_y[j]
         new_y[j] = old_y[j] + alpha * old_y[i]

       using temporaries to hold old_y[i] and old_y[j]. The central element
       (when k is odd) is updated as:
         new_y[mid] = old_y[mid] + alpha * old_y[mid]
                    = (1 + alpha) * old_y[mid]

       This preserves exact semantics while improving cache usage and eliminating
       the O(N) temporary array z. */

    int left  = 0;
    int right = k - 1;
    const DATA_TYPE alpha_local = alpha;

    /* Pairwise symmetric update: (left, right) with left < right. */
    while (left < right) {
      const DATA_TYPE y_left  = y[left];
      const DATA_TYPE y_right = y[right];

      y[left]  = y_left  + alpha_local * y_right;
      y[right] = y_right + alpha_local * y_left;

      ++left;
      --right;
    }

    /* Handle the middle element when k is odd (left == right). */
    if (left == right) {
      const DATA_TYPE y_mid = y[left];
      y[left] = y_mid * (SCALAR_VAL(1.0) + alpha_local);
    }

    /* Last element is set to alpha. */
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