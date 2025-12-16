/**
 * Exo Cholesky driver: mirrors PolyBench/C cholesky.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"

/* Include the Exo-generated kernel header. */
#include "generated/cholesky/cholesky.h"


/* Array initialization. */
static
void init_array(int n,
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
    for (j = 0; j <= i; j++) {
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
from exo.API_scheduling import *  # imported as per guidelines (not used directly here)
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Extern that lowers to the PolyBench SQRT_FUN macro.
# This keeps the same numerical behavior as the original C kernel
# (sqrt vs sqrtf, etc. is selected by PolyBench).
class _SqrtFun(Extern):
    def __init__(self):
        super().__init__("sqrt_fun")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        arg_type = args[0].type
        if not arg_type.is_real_scalar():
            raise _EErr(
                f"expected argument to be a real scalar value, but got type {arg_type}"
            )
        return arg_type

    def compile(self, args, prim_type):
        # SQRT_FUN comes from polybench/cholesky headers.
        return f"SQRT_FUN(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # Ensure SQRT_FUN macro is visible in the generated C file.
        return '#include "cholesky.h"'

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_fun = _SqrtFun()


@proc
def kernel_cholesky(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
):
    # Scratch buffer storing reciprocals of diagonal entries:
    #   inv_diag[j] = 1 / A[j, j]   once row j is finished.
    # This turns O(n^2) divisions in the original kernel into
    # O(n^2) multiplications + O(n) divisions.
    inv_diag: DATA_TYPE[n] @ DRAM

    for i in seq(0, n):
        # ------------------------------------------------------------
        # Off-diagonal elements in row i: j = 0 .. i-1
        # Original code:
        #   for j in 0..i-1:
        #     for k in 0..j-1: A[i,j] -= A[i,k] * A[j,k]
        #     A[i,j] = A[i,j] / A[j,j]
        #
        # We compute the inner dot product with a scalar accumulator
        #   s_ij = sum_{k=0}^{j-1} A[i,k] * A[j,k]
        # and then perform a single update:
        #   A[i,j] = (A[i,j] - s_ij) * inv_diag[j]
        #
        # This:
        #   * keeps A[i,j] in a register (only one load / one store),
        #   * replaces the division by a multiply with cached 1 / A[j,j].
        # ------------------------------------------------------------
        for j in seq(0, i):
            s_ij: DATA_TYPE
            s_ij = 0

            # Unroll the k-loop by a factor of 4 to expose ILP and
            # help the backend vectorizer. The expression (j / 4) * 4
            # is the largest multiple of 4 less than or equal to j.
            for k in seq(0, (j / 4) * 4):
                s_ij += A[i, k] * A[j, k]
                s_ij += A[i, k + 1] * A[j, k + 1]
                s_ij += A[i, k + 2] * A[j, k + 2]
                s_ij += A[i, k + 3] * A[j, k + 3]

            # Handle the remaining k values (0–3 iterations).
            for k in seq((j / 4) * 4, j):
                s_ij += A[i, k] * A[j, k]

            # At this point, for all j < i we have already computed
            # the diagonal A[j,j] and its reciprocal inv_diag[j].
            A[i, j] = (A[i, j] - s_ij) * inv_diag[j]

        # ------------------------------------------------------------
        # Diagonal element A[i, i]
        # Original code:
        #   for k in 0..i-1: A[i,i] -= A[i,k] * A[i,k]
        #   A[i,i] = SQRT_FUN(A[i,i])
        #
        # We compute:
        #   s_ii = sum_{k=0}^{i-1} A[i,k]^2
        #   A[i,i] = sqrt_fun(A[i,i] - s_ii)
        #
        # Again, A[i,i] is only loaded and stored once, and the
        # inner-product is accumulated in a scalar s_ii.
        # ------------------------------------------------------------
        s_ii: DATA_TYPE
        s_ii = 0

        # Unrolled k-loop by factor 4, same pattern as above.
        for k in seq(0, (i / 4) * 4):
            s_ii += A[i, k] * A[i, k]
            s_ii += A[i, k + 1] * A[i, k + 1]
            s_ii += A[i, k + 2] * A[i, k + 2]
            s_ii += A[i, k + 3] * A[i, k + 3]

        for k in seq((i / 4) * 4, i):
            s_ii += A[i, k] * A[i, k]

        A[i, i] = sqrt_fun(A[i, i] - s_ii)

        # Cache 1 / A[i,i] for use in subsequent rows (i' > i).
        # This single division is reused many times when updating
        # off-diagonal entries in later rows.
        inv_diag[i] = 1.0 / A[i, i]
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

  /* Run Exo kernel. Flatten 2D view to a 1D pointer. */
  kernel_cholesky(/*ctxt=*/NULL,
                  n,
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