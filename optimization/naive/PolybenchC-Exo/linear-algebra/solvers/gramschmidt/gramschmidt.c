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
from exo.API_scheduling import *          # scheduling primitives
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Extern for sqrt, mapped to C's sqrt from <math.h>.
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
        # Return the same real-scalar type as the input
        return arg_type

    def compile(self, args, prim_type):
        # Emit a call to C sqrt, with explicit cast to the primitive type.
        return f"sqrt(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # Ensure the math header is available for sqrt.
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_exo = _Sqrt()


# Baseline Gram-Schmidt kernel: faithful to the PolyBench reference.
@proc
def kernel_gramschmidt_unopt(
    m: size,
    n: size,
    A: DATA_TYPE[m, n] @ DRAM,
    R: DATA_TYPE[n, n] @ DRAM,
    Q: DATA_TYPE[m, n] @ DRAM,
):
    nrm: DATA_TYPE

    for k in seq(0, n):
        # nrm = sum_{i=0}^{m-1} A[i, k]^2
        nrm = 0.0
        for i in seq(0, m):
            nrm += A[i, k] * A[i, k]

        # R[k, k] = sqrt(nrm)
        R[k, k] = sqrt_exo(nrm)

        # Q[:, k] = A[:, k] / R[k, k]
        for i in seq(0, m):
            Q[i, k] = A[i, k] / R[k, k]

        # For each j > k, update R[k, j] and A[:, j]
        for j in seq(k + 1, n):
            # R[k, j] = Q[:, k]^T * A[:, j]
            R[k, j] = 0.0
            for i in seq(0, m):
                R[k, j] += Q[i, k] * A[i, j]

            # A[:, j] = A[:, j] - Q[:, k] * R[k, j]
            for i in seq(0, m):
                A[i, j] = A[i, j] - Q[i, k] * R[k, j]


# Scheduling / optimization:
# - We stage the k-th column of A into a contiguous buffer A_col for the
#   entire body of the k-loop. This lets us reuse A[:, k] across the norm
#   and Q-formation loops while reading each element only once from DRAM.
# - We stage the k-th column of Q into a contiguous buffer Q_col for the
#   j-loop nest. This converts heavily reused, strided accesses Q[i, k]
#   into dense accesses Q_col[i], improving cache locality and vectorization.
kernel_gramschmidt = rename(kernel_gramschmidt_unopt, "kernel_gramschmidt")

# Stage A[:, k] for the whole body of the k-loop.
k_loop = kernel_gramschmidt.find_loop("for k in _:_" )
kernel_gramschmidt = stage_mem(
    kernel_gramschmidt,
    k_loop.body(),          # entire body of the k-loop
    "A[0:m, k]",            # window: k-th column of A
    "A_col",                # new staging buffer
)

# Re-find the k-loop after the transformation (AST has changed).
k_loop = kernel_gramschmidt.find_loop("for k in _:_" )

# Stage Q[:, k] only around the j-loop nest where it is heavily reused.
j_loop = kernel_gramschmidt.find_loop("for j in _:_" )
kernel_gramschmidt = stage_mem(
    kernel_gramschmidt,
    j_loop.as_block(),      # just the j-loop and its body
    "Q[0:m, k]",            # window: k-th column of Q
    "Q_col",                # new staging buffer
)

# Simplify to clean up any redundant loads/stores introduced by staging.
kernel_gramschmidt = simplify(kernel_gramschmidt)
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