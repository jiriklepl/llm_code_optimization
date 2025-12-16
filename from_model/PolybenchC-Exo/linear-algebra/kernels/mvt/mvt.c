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

# Base kernel: two matrix–vector products
#   x1[i] += sum_j A[i,j] * y_1[j]
#   x2[i] += sum_j A[j,i] * y_2[j]
# The second pass is rewritten algebraically to traverse A row‑major
# for better cache locality:
#   x2[i] = x2[i] + sum_j A[j,i] * y_2[j]
# is equivalent to
#   for j:
#     tmp = y_2[j]
#     for i:
#       x2[i] += A[j,i] * tmp
#
# We also use scalar accumulators for x1[i] to reduce repeated loads/stores.

@proc
def kernel_mvt_base(
    n: size,
    x1: DATA_TYPE[n] @ DRAM,
    x2: DATA_TYPE[n] @ DRAM,
    y_1: DATA_TYPE[n] @ DRAM,
    y_2: DATA_TYPE[n] @ DRAM,
    A: DATA_TYPE[n, n] @ DRAM,
):

    # Phase 1: x1 += A * y_1, with a scalar accumulator per row i.
    for i in seq(0, n):
        acc1: DATA_TYPE
        acc1 = x1[i]
        for j in seq(0, n):
            acc1 += A[i, j] * y_1[j]
        x1[i] = acc1

    # Phase 2: x2 += A^T * y_2, but traversing A row‑major.
    # For each row j, we reuse y_2[j] via a scalar and sweep i
    # contiguously over that row.
    for j in seq(0, n):
        y2_j: DATA_TYPE
        y2_j = y_2[j]
        for i in seq(0, n):
            x2[i] = x2[i] + A[j, i] * y2_j


# -----------------------------
# Scheduling / optimization
# -----------------------------
# We now apply Exo scheduling primitives to improve parallelism and
# data locality while preserving the above semantics.

p = kernel_mvt_base

# 1) Tile the outer i-loop of the first phase to form coarse-grain
#    row blocks and then parallelize across those tiles.
#
#    Original (phase 1):
#      for i in 0..n-1:
#        ...
#
#    After divide_loop with div_const = 64:
#      for io in 0..ceil(n/64)-1:
#        for ii in 0..63:
#          i = 64*io + ii   (with guards for out-of-bounds iterations)
#
#    Each tile 'io' owns a disjoint slice of x1, so tiles can be
#    executed in parallel safely.
p = divide_loop(p, "i", 64, ("io", "ii"), tail="guard")

# Parallelize the outer tile loop of phase 1 across cores.
p = parallelize_loop(p, p.find_loop("io"))

# 2) Parallelize the inner i-loop of the second phase.
#
#    Phase 2 loop structure is:
#      for j in 0..n-1:
#        y2_j = y_2[j]
#        for i in 0..n-1:
#          x2[i] += A[j, i] * y2_j
#
#    For a fixed j, iterations over i update distinct elements x2[i],
#    so the i-loop is safely parallelizable. Different j-iterations
#    remain sequential, preserving the reduction over j for each x2[i].
p = parallelize_loop(p, p.find_loop("i"))

# 3) Export the scheduled kernel under the name expected by the C driver.
kernel_mvt = rename(p, "kernel_mvt")
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