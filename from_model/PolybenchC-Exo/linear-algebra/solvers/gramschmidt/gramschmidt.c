/**
 * Exo Gram-Schmidt driver: mirrors PolyBench/C gramschmidt.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"

/* Include the Exo-generated kernel header. */
#include "generated/gramschmidt/gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *  # scheduling primitives (not used directly, but imported per guidelines)
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Extern for sqrt, lowered to the C math library.
class _Sqrt(Extern):
    def __init__(self):
        super().__init__("sqrt")

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
        # Emit a call to the C sqrt function with an explicit cast
        return f"sqrt(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # Ensure the math header is available for sqrt
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_exo = _Sqrt()

# Tiling factors chosen to improve cache locality while keeping code simple.
# They are purely compile-time constants from Exo's perspective.
TILE_I = 64  # tile size in the row (i) dimension
TILE_J = 64  # tile size in the trailing-column (j) dimension


@proc
def kernel_gramschmidt(
    m: size,
    n: size,
    A: DATA_TYPE[m, n] @ DRAM,
    R: DATA_TYPE[n, n] @ DRAM,
    Q: DATA_TYPE[m, n] @ DRAM,
):
    # Temporary accumulator for the squared 2-norm of column k
    nrm: DATA_TYPE
    # Per-row scalar used to hold Q[i, k] while processing tiles
    qik: DATA_TYPE

    for k in seq(0, n):
        # ------------------------------------------------------------
        # 1. Compute the squared 2-norm of column k of the *current* A:
        #       nrm = sum_{i=0}^{m-1} A[i, k]^2
        # ------------------------------------------------------------
        nrm = 0.0
        for i in seq(0, m):
            nrm += A[i, k] * A[i, k]

        # ------------------------------------------------------------
        # 2. Set the diagonal of R and initialize the strict upper part
        #    of row k. R[k, k] = sqrt(nrm), and R[k, j] = 0 for j > k
        #    before accumulation.
        # ------------------------------------------------------------
        R[k, k] = sqrt_exo(nrm)

        for j in seq(k + 1, n):
            R[k, j] = 0.0

        # ------------------------------------------------------------
        # 3. Fused normalization of Q[:, k] and accumulation of the
        #    k-th row of R for all j > k.
        #
        #    For each row i:
        #       qik = A[i, k] / R[k, k]
        #       Q[i, k] = qik
        #       For each j > k:
        #           R[k, j] += qik * A[i, j]
        #
        #    We implement this with 2D tiling in i and j to improve
        #    cache locality. The loop order still corresponds to:
        #       for i in 0..m-1:
        #           for j in k+1..n-1:
        #               R[k, j] += Q[i, k] * A[i, j]
        #    so the per-(k, j) accumulation order over i is preserved.
        # ------------------------------------------------------------
        for I in seq(0, (m + TILE_I - 1) / TILE_I):
            for i in seq(0, TILE_I):
                if TILE_I * I + i < m:
                    # Global row index
                    # Normalize A[row, k] to get Q[row, k].
                    qik = A[TILE_I * I + i, k] / R[k, k]
                    Q[TILE_I * I + i, k] = qik

                    # Accumulate contributions to all trailing columns j > k.
                    for J in seq(0, (n - (k + 1) + TILE_J - 1) / TILE_J):
                        for j in seq(0, TILE_J):
                            if (k + 1) + TILE_J * J + j < n:
                                R[k, (k + 1) + TILE_J * J + j] += \
                                    qik * A[TILE_I * I + i, (k + 1) + TILE_J * J + j]

        # ------------------------------------------------------------
        # 4. Rank-1 update of the trailing submatrix:
        #       A[:, j] <- A[:, j] - Q[:, k] * R[k, j]  for all j > k
        #
        #    Again we traverse with (i outer, j inner) plus tiling to
        #    align with row-major layout. The update order per element
        #    (i, j) matches the reference algorithm.
        # ------------------------------------------------------------
        for I in seq(0, (m + TILE_I - 1) / TILE_I):
            for i in seq(0, TILE_I):
                if TILE_I * I + i < m:
                    qik = Q[TILE_I * I + i, k]

                    for J in seq(0, (n - (k + 1) + TILE_J - 1) / TILE_J):
                        for j in seq(0, TILE_J):
                            if (k + 1) + TILE_J * J + j < n:
                                A[TILE_I * I + i, (k + 1) + TILE_J * J + j] = \
                                    A[TILE_I * I + i, (k + 1) + TILE_J * J + j] - \
                                    qik * R[k, (k + 1) + TILE_J * J + j]
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_gramschmidt (/*ctxt=*/NULL, m, n,
		      (DATA_TYPE*)POLYBENCH_ARRAY(A),
		      (DATA_TYPE*)POLYBENCH_ARRAY(R),
		      (DATA_TYPE*)POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}