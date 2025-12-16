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


@proc
def kernel_ludcmp(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
    b: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    w: DATA_TYPE

    # LU decomposition
    for i in seq(0, n):
        # Compute L entries (below diagonal)
        for j in seq(0, i):
            w = A[i, j]
            for k in seq(0, j):
                w = w - A[i, k] * A[k, j]
            A[i, j] = w / A[j, j]

        # Compute U entries (on and above diagonal)
        for j in seq(i, n):
            w = A[i, j]
            for k in seq(0, i):
                w = w - A[i, k] * A[k, j]
            A[i, j] = w

    # Forward substitution to solve L * y = b
    for i in seq(0, n):
        w = b[i]
        for j in seq(0, i):
            w = w - A[i, j] * y[j]
        y[i] = w

    # Backward substitution to solve U * x = y
    # Original C loop:
    # for (i = n-1; i >= 0; i--) {
    #   w = y[i];
    #   for (j = i+1; j < n; j++)
    #       w -= A[i][j] * x[j];
    #   x[i] = w / A[i][i];
    # }
    #
    # Rewritten in forward iteration form with equivalent semantics.
    for i in seq(0, n):
        # Corresponds to original index i0 = n - 1 - i
        w = y[n - 1 - i]
        for j in seq(n - i, n):
            w = w - A[n - 1 - i, j] * x[j]
        x[n - 1 - i] = w / A[n - 1 - i, n - 1 - i]
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