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


# Baseline LU decomposition + forward/backward substitution.
@proc
def kernel_ludcmp_base(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
    b: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    w: DATA_TYPE

    # LU decomposition in-place on A.
    for i in seq(0, n):
        # Compute L entries (below diagonal): A[i, jl]
        for jl in seq(0, i):
            w = A[i, jl]
            for kl in seq(0, jl):
                w = w - A[i, kl] * A[kl, jl]
            A[i, jl] = w / A[jl, jl]

        # Compute U entries (on and above diagonal): A[i, ju]
        for ju in seq(i, n):
            w = A[i, ju]
            for ku in seq(0, i):
                w = w - A[i, ku] * A[ku, ju]
            A[i, ju] = w

    # Forward substitution to solve L * y = b.
    for ifs in seq(0, n):
        w = b[ifs]
        for jf in seq(0, ifs):
            w = w - A[ifs, jf] * y[jf]
        y[ifs] = w

    # Backward substitution to solve U * x = y.
    # This is written as a forward loop over ib with a reversed index n-1-ib.
    for ib in seq(0, n):
        w = y[n - 1 - ib]
        for jb in seq(n - ib, n):
            w = w - A[n - 1 - ib, jb] * x[jb]
        x[n - 1 - ib] = w / A[n - 1 - ib, n - 1 - ib]


# ---------------------------------------------------------------------------
# Scheduling for performance
# - Unroll the innermost kl and ku loops in the LU factorization.
# - Unroll the inner jf loop in the forward substitution.
# These transforms increase instruction-level parallelism and give
# the backend compiler more opportunities to vectorize, while
# preserving the original execution order.
# ---------------------------------------------------------------------------

kernel_ludcmp_opt = kernel_ludcmp_base

# Unroll the kl loop in the L update.
kernel_ludcmp_opt = divide_loop(kernel_ludcmp_opt, "kl", 4, ("kl_outer", "kl_inner"), tail="guard")
kernel_ludcmp_opt = unroll_loop(kernel_ludcmp_opt, "kl_inner")

# Unroll the ku loop in the U update.
kernel_ludcmp_opt = divide_loop(kernel_ludcmp_opt, "ku", 4, ("ku_outer", "ku_inner"), tail="guard")
kernel_ludcmp_opt = unroll_loop(kernel_ludcmp_opt, "ku_inner")

# Unroll the jf loop in the forward substitution.
kernel_ludcmp_opt = divide_loop(kernel_ludcmp_opt, "jf", 4, ("jf_outer", "jf_inner"), tail="guard")
kernel_ludcmp_opt = unroll_loop(kernel_ludcmp_opt, "jf_inner")

# Clean up the resulting IR.
kernel_ludcmp_opt = simplify(kernel_ludcmp_opt)

# Export the scheduled kernel under the name expected by the C driver.
kernel_ludcmp = rename(kernel_ludcmp_opt, "kernel_ludcmp")
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