/**
 * Exo GEMVER driver: mirrors PolyBench/C gemver.c but calls the Exo-generated kernel.
 *
 * Original authors:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

/* Include the Exo-generated kernel header. */
#include "generated/gemver/gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Tunable blocking parameters (chosen for a modern multicore x86-64 CPU).
# These affect traversal order and parallel grain size but do not change
# the mathematical result.
TI_A: int = 32   # tile size along i for the A update
TJ_A: int = 32   # tile size along j for the A update
TI_X: int = 32   # tile size along i for the x update
TI_W: int = 32   # tile size along i for the w update
TJ_W: int = 32   # tile size along j for the w update


@proc
def kernel_gemver_impl(
    n: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    A: DATA_TYPE[n, n] @ DRAM,
    u1: DATA_TYPE[n] @ DRAM,
    v1: DATA_TYPE[n] @ DRAM,
    u2: DATA_TYPE[n] @ DRAM,
    v2: DATA_TYPE[n] @ DRAM,
    w: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
    z: DATA_TYPE[n] @ DRAM,
):
    #
    # Phase 1: rank-2 update on A
    #   A := A + u1 v1^T + u2 v2^T
    # Row-major A, so we keep j (column) as the innermost loop.
    #
    for i in seq(0, n):
        for j in seq(0, n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    #
    # Phase 2: x := x + beta * A^T y
    # Original PolyBench/Exo structure:
    #   for i:
    #       for j:
    #           x[i] += beta * A[j, i] * y[j]
    #
    # This directly matches the original semantics; we will later
    # parallelize across i while keeping this simple, clear form.
    #
    for i in seq(0, n):
        for j in seq(0, n):
            x[i] += beta * A[j, i] * y[j]

    #
    # Phase 3: x := x + z
    # Simple elementwise SAXPY-like update.
    #
    for i in seq(0, n):
        x[i] += z[i]

    #
    # Phase 4: w := w + alpha * A x
    # Standard GEMV:
    #   for i:
    #     for j:
    #       w[i] += alpha * A[i, j] * x[j]
    # Row-major A, so again keep j innermost for unit-stride access.
    #
    for i in seq(0, n):
        for j in seq(0, n):
            w[i] += alpha * A[i, j] * x[j]


# Start from the clear specification above and apply a performance-oriented
# schedule. All transformations below preserve the mathematical behavior;
# they only change loop nesting, tiling, and parallelization.

kernel_gemver_sched = kernel_gemver_impl

# ------------------------------------------------------------
# Phase 1: tile and parallelize the A update over (i, j)
# ------------------------------------------------------------
# - 2D tiling (i by TI_A, j by TJ_A) improves cache locality on A.
# - io_A is an outer loop over row tiles and is safe to parallelize
#   because each tile updates disjoint rows of A.
kernel_gemver_sched = divide_loop(kernel_gemver_sched, "i", TI_A, ["io_A", "ii_A"])
kernel_gemver_sched = divide_loop(kernel_gemver_sched, "j", TJ_A, ["jo_A", "ji_A"])
# Reorder the tile loops so that we iterate tiles in (io_A, jo_A) order,
# with (ii_A, ji_A) providing the fine-grain iteration inside each tile.
kernel_gemver_sched = reorder_loops(kernel_gemver_sched, "ii_A jo_A")
# Parallelize across row-tiles of A; no cross-tile write conflicts.
kernel_gemver_sched = parallelize_loop(kernel_gemver_sched, "io_A")

# ------------------------------------------------------------
# Phase 2: tile and parallelize the x update over i
# ------------------------------------------------------------
# Each i-iteration updates a distinct x[i] and reads a full column of A
# and the whole vector y. We tile i to control parallel grain size and
# parallelize over tiles.
kernel_gemver_sched = divide_loop(kernel_gemver_sched, "i", TI_X, ["io_X", "ii_X"])
kernel_gemver_sched = parallelize_loop(kernel_gemver_sched, "io_X")

# ------------------------------------------------------------
# Phase 3: x := x + z (elementwise)
# ------------------------------------------------------------
# Independent per element, so we simply parallelize the i-loop.
kernel_gemver_sched = parallelize_loop(kernel_gemver_sched, "i")

# ------------------------------------------------------------
# Phase 4: tile and parallelize the final w update
# ------------------------------------------------------------
# Similar to Phase 1, but now each row i accumulates a dot-product
# w[i] += alpha * sum_j A[i, j] * x[j].
# We tile rows with TI_W and columns with TJ_W to improve reuse of A and x,
# and parallelize across row tiles.
kernel_gemver_sched = divide_loop(kernel_gemver_sched, "i #1", TI_W, ["io_W", "ii_W"])
kernel_gemver_sched = divide_loop(kernel_gemver_sched, "j #1", TJ_W, ["jo_W", "ji_W"])
kernel_gemver_sched = parallelize_loop(kernel_gemver_sched, "io_W")

# Expose the final scheduled kernel under the name expected by the C driver.
kernel_gemver = kernel_gemver_sched
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z,  DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to raw pointers. */
  kernel_gemver(/*ctxt=*/NULL, n,
                (DATA_TYPE*)&alpha,
                (DATA_TYPE*)&beta,
                (DATA_TYPE*)POLYBENCH_ARRAY(A),
                (DATA_TYPE*)POLYBENCH_ARRAY(u1),
                (DATA_TYPE*)POLYBENCH_ARRAY(v1),
                (DATA_TYPE*)POLYBENCH_ARRAY(u2),
                (DATA_TYPE*)POLYBENCH_ARRAY(v2),
                (DATA_TYPE*)POLYBENCH_ARRAY(w),
                (DATA_TYPE*)POLYBENCH_ARRAY(x),
                (DATA_TYPE*)POLYBENCH_ARRAY(y),
                (DATA_TYPE*)POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}