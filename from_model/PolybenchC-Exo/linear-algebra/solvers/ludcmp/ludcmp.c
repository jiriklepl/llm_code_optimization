/**
 * Exo LUDCMP driver: mirrors PolyBench/C ludcmp.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

/* Include the Exo-generated kernel header. */
#include "generated/ludcmp/ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *
from exo.API_cursors import ForCursor


@proc
def kernel_ludcmp(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
    b: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    # Temporary scalar accumulator
    w: DATA_TYPE
    # Scratch array holding 1 / A[i, i] for each pivot row i.
    # Size is O(n) << O(n^2) of A, so the extra memory is negligible.
    inv_diag: DATA_TYPE[n] @ DRAM

    # LU decomposition (in-place, no pivoting).
    # We keep the original update formulas but improve the U update's
    # memory access pattern by interchanging the (j, k) loops. This
    # makes accesses to A[k, j] contiguous in j (row-major friendly)
    # while preserving the per-element update order.
    for i in seq(0, n):
        # Compute L entries (strictly below diagonal).
        # Original C:
        #   w = A[i][j];
        #   for (k = 0; k < j; k++) w -= A[i][k] * A[k][j];
        #   A[i][j] = w / A[j][j];
        for j in seq(0, i):
            w = A[i, j]
            for k in seq(0, j):
                w = w - A[i, k] * A[k, j]
            A[i, j] = w / A[j, j]

        # Compute U entries (on and above diagonal) with improved locality.
        #
        # Original C / Exo form:
        #   for j in seq(i, n):
        #       w = A[i, j]
        #       for k in seq(0, i):
        #           w = w - A[i, k] * A[k, j]
        #       A[i, j] = w
        #
        # This is algebraically equivalent to:
        #   for k in seq(0, i):
        #       for j in seq(i, n):
        #           A[i, j] = A[i, j] - A[i, k] * A[k, j]
        #
        # For each fixed (i, j), the sequence of updates to A[i, j]
        # as k increases is unchanged; we only interleave independent
        # j-updates. That preserves floating-point semantics while
        # making A[k, j] accesses unit-stride in j.
        for k in seq(0, i):
            for j in seq(i, n):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]

        # Precompute and cache the reciprocal of the pivot on this row.
        # This will be reused in the backward substitution phase to
        # replace a division by a multiplication.
        inv_diag[i] = 1.0 / A[i, i]

    # Forward substitution to solve L * y = b.
    # L has unit diagonal and is stored in the strict lower part of A.
    for i2 in seq(0, n):
        w = b[i2]
        for j in seq(0, i2):
            w = w - A[i2, j] * y[j]
        y[i2] = w

    # Backward substitution to solve U * x = y.
    #
    # Original C loop:
    #   for (i = n-1; i >= 0; i--) {
    #     w = y[i];
    #     for (j = i+1; j < n; j++)
    #         w -= A[i][j] * x[j];
    #     x[i] = w / A[i][i];
    #   }
    #
    # We express this using a forward loop index ii with the same
    # effective iteration order on the physical index (n-1-ii).
    # We also use inv_diag to replace the final division by a multiply.
    for ii in seq(0, n):
        # Corresponds to original index i0 = n - 1 - ii
        w = y[n - 1 - ii]
        for j in seq(n - ii, n):
            w = w - A[n - 1 - ii, j] * x[j]
        x[n - 1 - ii] = w * inv_diag[n - 1 - ii]


# --------------------------------------------------------------------
# Scheduling / optimization
# --------------------------------------------------------------------
# We now apply a small amount of scheduling to expose thread-level
# parallelism in the U-update j-loop (which has no loop-carried
# dependencies across j for a fixed (i, k)). The analysis in Exo
# will reject unsafe parallelizations automatically.
# --------------------------------------------------------------------

kernel_ludcmp_sched = kernel_ludcmp

# Find the j-loop inside the U update whose parent loop variable is `k`.
# This is the loop:
#   for k in seq(0, i):
#       for j in seq(i, n):
#           A[i, j] = A[i, j] - A[i, k] * A[k, j]
#
# For each fixed (i, k), iterations over j are independent, so this
# inner loop can be parallelized.
j_loops = kernel_ludcmp_sched.find_all("for j in _:_")
for jl in j_loops:
    parent = jl.parent()
    if isinstance(parent, ForCursor) and parent.name() == "k":
        kernel_ludcmp_sched = parallelize_loop(kernel_ludcmp_sched, jl)
        break

# Optionally perform some algebraic simplifications after annotation.
kernel_ludcmp_sched = simplify(kernel_ludcmp_sched)

# Rename the scheduled version back to the original exported name so
# the C driver continues to call `kernel_ludcmp`.
kernel_ludcmp = rename(kernel_ludcmp_sched, "kernel_ludcmp")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten views to raw pointers. */
  kernel_ludcmp(/*ctxt=*/NULL, n,
                (DATA_TYPE*)POLYBENCH_ARRAY(A),
                (DATA_TYPE*)POLYBENCH_ARRAY(b),
                (DATA_TYPE*)POLYBENCH_ARRAY(x),
                (DATA_TYPE*)POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}