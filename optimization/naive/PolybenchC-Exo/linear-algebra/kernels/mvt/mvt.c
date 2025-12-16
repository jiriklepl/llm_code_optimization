/**
 * Exo MVT driver: mirrors PolyBench/C mvt.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"

/* Include the Exo-generated kernel header. */
#include "generated/mvt/mvt.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i]  = (DATA_TYPE) (i % n)      / n;
      x2[i]  = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i * j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Baseline matrix-vector transpose kernel in Exo.
# We follow the original PolyBench semantics:
#   x1[i] += sum_j A[i,j] * y_1[j]
#   x2[i] += sum_j A[j,i] * y_2[j]
@proc
def kernel_mvt_base(
    n: size,
    x1: DATA_TYPE[n] @ DRAM,
    x2: DATA_TYPE[n] @ DRAM,
    y_1: DATA_TYPE[n] @ DRAM,
    y_2: DATA_TYPE[n] @ DRAM,
    A: DATA_TYPE[n, n] @ DRAM,
):
    # First matrix-vector product: x1 = x1 + A * y_1
    for i in seq(0, n):
        for j in seq(0, n):
            # Use a reduction statement to make the accumulation explicit.
            x1[i] += A[i, j] * y_1[j]

    # Second matrix-vector product: x2 = x2 + A^T * y_2
    for i in seq(0, n):
        for j in seq(0, n):
            x2[i] += A[j, i] * y_2[j]


# ---------------------------------------------------------------------------
# Scheduling: optimize kernel_mvt_base into kernel_mvt.
#
# Goals:
#  - Improve cache locality and enable vectorization by blocking the
#    inner j loops in both matrix-vector products.
#  - Unroll the small j tiles to expose instruction-level parallelism.
#  - Parallelize the outer i loops, which are provably independent
#    (each iteration only updates x1[i] or x2[i]).
# ---------------------------------------------------------------------------

kernel_mvt = kernel_mvt_base

# Tile the inner j loop of the first matrix-vector multiply.
# We use a tile size of 32; the 'guard' tail strategy safely handles
# the case where n is not a multiple of 32.
kernel_mvt = divide_loop(kernel_mvt, "j", 32, ("j0_outer", "j0_inner"), tail="guard")

# After the previous step, the only remaining loop with iterator name 'j'
# is the inner loop of the second matrix-vector multiply. Tile it as well.
kernel_mvt = divide_loop(kernel_mvt, "j", 32, ("j1_outer", "j1_inner"), tail="guard")

# Unroll the small j tiles (each has constant trip-count = 32) for both
# matrix-vector products. This gives the C compiler more opportunity to
# vectorize and improves instruction-level parallelism.
kernel_mvt = unroll_loop(kernel_mvt, "j0_inner")
kernel_mvt = unroll_loop(kernel_mvt, "j1_inner")

# Parallelize the outer i loops of both products.
# These loops are independent because each iteration touches a distinct
# element of x1 or x2 and only reads from A, y_1, and y_2.
kernel_mvt = parallelize_loop(kernel_mvt, "i")      # outer i-loop of first product
kernel_mvt = parallelize_loop(kernel_mvt, "i #1")   # outer i-loop of second product

EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,   DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to 1D pointers. */
  kernel_mvt(/*ctxt=*/NULL, n,
             (DATA_TYPE*)POLYBENCH_ARRAY(x1),
             (DATA_TYPE*)POLYBENCH_ARRAY(x2),
             (DATA_TYPE*)POLYBENCH_ARRAY(y_1),
             (DATA_TYPE*)POLYBENCH_ARRAY(y_2),
             (DATA_TYPE*)POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n,
                                    POLYBENCH_ARRAY(x1),
                                    POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}