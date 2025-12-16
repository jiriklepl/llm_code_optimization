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
from exo.libs.memories import DRAM

@proc
def kernel_durbin(
    n: size,
    r: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    z: DATA_TYPE[n] @ DRAM
    alpha: DATA_TYPE
    beta: DATA_TYPE
    sum: DATA_TYPE

    # Initial conditions
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]

    for k in seq(1, n):
        beta = (1.0 - alpha * alpha) * beta
        sum = 0.0

        for i in seq(0, k):
            sum += r[k - i - 1] * y[i]

        alpha = -(r[k] + sum) / beta

        for i in seq(0, k):
            z[i] = y[i] + alpha * y[k - i - 1]

        for i in seq(0, k):
            y[i] = z[i]

        y[k] = alpha
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