/**
 * Exo DURBIN driver: mirrors PolyBench/C durbin.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"

/* Include the Exo-generated kernel header. */
#include "generated/durbin/durbin.h"


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

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

@proc
def kernel_durbin(
    n: size,
    r: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    # Working scalars for the Durbin recursion
    alpha: DATA_TYPE
    beta: DATA_TYPE
    sum: DATA_TYPE

    # The algorithm requires at least one element.
    assert n >= 1

    # Initial conditions (k = 0)
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]

    # Main Levinson–Durbin / Durbin recursion for k = 1..n-1
    #
    # This implementation is mathematically equivalent to the original
    # reference Exo kernel, but it eliminates the temporary buffer z[n]
    # by performing an in-place symmetric update of y[0..k-1].
    #
    # Original update (for each k):
    #   for i in 0..k-1:
    #       z[i] = y[i] + alpha * y[k-1-i]
    #   for i in 0..k-1:
    #       y[i] = z[i]
    #
    # This sets, for all i in [0, k-1]:
    #   y_new[i] = y_old[i] + alpha * y_old[k-1-i].
    #
    # The in-place version below updates disjoint index pairs
    # (i, j = k-1-i) using *only* the old values y_old[i], y_old[j]:
    #
    #   yi = y[i]; yj = y[j]
    #   y[i] = yi + alpha * yj
    #   y[j] = yj + alpha * yi
    #
    # For odd k, when i == j (the middle element), both assignments
    # write the same value y[i] = (1 + alpha) * y_old[i], exactly
    # matching the original two-pass z-based formulation.
    #
    # This reduces memory traffic (one pass instead of two, and no z)
    # while preserving the exact mathematical result.
    for k in seq(1, n):
        # Update beta using the reflection coefficient from the previous step.
        beta = (1.0 - alpha * alpha) * beta

        # Dot-product-like reduction:
        #   sum = Σ_{i=0}^{k-1} r[k-1-i] * y[i]
        # Uses y[0..k-1] from the previous k-step.
        sum = 0.0
        for i in seq(0, k):
            sum += r[k - i - 1] * y[i]

        # New reflection coefficient alpha_k.
        alpha = -(r[k] + sum) / beta

        # In-place symmetric update of y[0..k-1].
        # Number of iterations is ceil(k/2) = (k + 1) / 2 in integer arithmetic.
        for i in seq(0, (k + 1) / 2):
            yi: DATA_TYPE
            yj: DATA_TYPE

            yi = y[i]
            yj = y[k - i - 1]

            y[i] = yi + alpha * yj
            y[k - i - 1] = yj + alpha * yi

        # Set y[k] = alpha_k
        y[k] = alpha

# Apply a light-weight cleanup/simplification pass. This is a scheduling
# directive that canonicalizes expressions and may expose more optimization
# opportunities to the backend (e.g., better constant folding).
kernel_durbin = simplify(kernel_durbin)
EXO END
*/


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

  /* Run Exo kernel. Flatten 1D views to raw pointers. */
  kernel_durbin(/*ctxt=*/NULL,
                n,
                (DATA_TYPE*)POLYBENCH_ARRAY(r),
                (DATA_TYPE*)POLYBENCH_ARRAY(y));

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