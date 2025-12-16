/**
 * Exo LU driver: mirrors PolyBench/C lu.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"

/* Include the Exo-generated kernel header. */
#include "generated/lu/lu.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

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
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM


# LU factorization kernel.
# We keep the standard Doolittle-style algorithm, but restructure the
# second (U) update to improve memory locality:
#
#   Original PolyBench kernel (U part, for fixed i):
#       for j in seq(i, n):
#           for k in seq(0, i):
#               A[i, j] -= A[i, k] * A[k, j]
#
#   Here we use k as the outer loop:
#       for k in seq(0, i):
#           aik = A[i, k]
#           for j in seq(i, n):
#               A[i, j] -= aik * A[k, j]
#
# For each (i, j) the sequence of updates over k is still in
# increasing k order, so the floating-point result for that entry
# is unchanged; we only interleave updates for different j's.
# This reordering:
#   - Reuses A[i, k] across the whole row segment j in [i, n),
#   - Accesses A[k, j] with unit stride in j (row-major),
#   - Exposes an innermost j-loop that the C compiler can easily
#     vectorize.


@proc
def kernel_lu_base(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
):
    for i in seq(0, n):

        # Compute L row i: j from 0 to i-1
        for j in seq(0, i):
            # A[i, j] -= sum_{k=0}^{j-1} A[i, k] * A[k, j]
            for k in seq(0, j):
                A[i, j] += -A[i, k] * A[k, j]

            # Divide by the diagonal element of U.
            A[i, j] = A[i, j] / A[j, j]

        # Compute U row i: j from i to n-1
        # Reordered as k-outer, j-inner for better locality.
        for k in seq(0, i):
            aik: DATA_TYPE
            aik = A[i, k]
            for j in seq(i, n):
                A[i, j] += -aik * A[k, j]


# Apply a light-weight simplification pass and then expose the
# optimized kernel under the name `kernel_lu`, which is what the
# C driver expects.
kernel_lu_simplified = simplify(kernel_lu_base)
kernel_lu = rename(kernel_lu_simplified, "kernel_lu")

EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D view to 1D pointer. */
  kernel_lu (/*ctxt=*/NULL, n,
             (DATA_TYPE*)POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}