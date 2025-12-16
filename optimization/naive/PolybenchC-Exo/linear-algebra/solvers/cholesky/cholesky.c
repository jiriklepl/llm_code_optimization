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
from exo.API_scheduling import *
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Wrap PolyBench's SQRT_FUN macro as an Extern so it can be used
# inside Exo expressions without putting data accesses in conditionals.
class _SqrtFun(Extern):
    def __init__(self):
        # Name as it appears in printed Exo code
        super().__init__("sqrt_fun")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        arg_type = args[0].type
        # Require a real scalar (matches PolyBench's SQRT_FUN usage)
        if not arg_type.is_real_scalar():
            raise _EErr(
                f"expected argument to be a real scalar value, but got type {arg_type}"
            )
        return arg_type

    def compile(self, args, prim_type):
        # Lower to the PolyBench SQRT_FUN macro (sqrt / sqrtf / sqrtl).
        # The cast ensures the right precision is used.
        return f"SQRT_FUN(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # Ensure SQRT_FUN is visible when compiling the generated kernel.
        # cholesky.h transitively pulls in polybench.h where SQRT_FUN is defined.
        return '#include "cholesky.h"'

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_fun = _SqrtFun()


# Baseline Cholesky factorization:
# A is symmetric positive-definite on entry, and we overwrite it in-place
# with its lower-triangular Cholesky factor L (upper part is ignored).
@proc
def cholesky_unoptimized(
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
):
    for i in seq(0, n):
        # Off-diagonal elements in row i (j < i)
        for j in seq(0, i):
            # Compute the dot product of row i and row j up to column j-1
            # and update A[i, j] in-place.
            for k in seq(0, j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A[i, j] = A[i, j] / A[j, j]

        # Diagonal element A[i, i]: subtract squared norm of the prefix
        for k in seq(0, i):
            A[i, i] = A[i, i] - A[i, k] * A[i, k]

        A[i, i] = sqrt_fun(A[i, i])


# --------------------
# Scheduling / optimization
# --------------------
# We now apply Exo scheduling primitives to improve performance while
# preserving the exact loop-carried dependencies of the Cholesky
# factorization.
#
# The main hotspot is the inner k-loop (dot products). We tile k into
# (ko, ki) so that each tile works on a contiguous chunk of the fastest
# varying dimension. This improves cache locality and gives the C
# compiler clearer opportunities for vectorization, without changing
# the k-iteration order (and thus preserving numerical behavior).


K_TILE = 32  # moderate tile size, works well on typical CPUs

# Tile the k loop in the off-diagonal update (j < i).
cholesky_tiled = divide_loop(
    cholesky_unoptimized,
    "k",               # first k-loop encountered (inside the j-loop)
    K_TILE,
    ("ko", "ki"),
    tail="guard",      # guards handle non-multiple-of-K_TILE sizes
)

# Tile the remaining k loop in the diagonal update.
cholesky_tiled = divide_loop(
    cholesky_tiled,
    "k",               # the remaining k-loop (inside the diagonal update)
    K_TILE,
    ("ko", "ki"),
    tail="guard",
)

# Clean up algebra and simplify guards where possible.
cholesky_tiled = simplify(cholesky_tiled)

# Rename to the externally visible name expected by the C driver.
kernel_cholesky = rename(cholesky_tiled, "kernel_cholesky")
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