/**
 * Exo SYRK driver: mirrors PolyBench/C syrk.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"

/* Include the Exo-generated kernel header. */
#include "generated/syrk/syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Base SYRK kernel:
#   C := beta * C + alpha * A * A^T
# Only the lower-triangular part of C (j <= i) is updated, matching PolyBench.
@proc
def kernel_syrk_base(
    n: size,
    m: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[n, n] @ DRAM,
    A: DATA_TYPE[n, m] @ DRAM,
):
    for i in seq(0, n):
        # Scale the lower-triangular part of row i by beta
        for j in seq(0, i + 1):
            C[i, j] = beta * C[i, j]

        # Rank-k update: alpha * A * A^T, lower triangle only
        for k in seq(0, m):
            for j in seq(0, i + 1):
                C[i, j] += alpha * A[i, k] * A[j, k]

# Create an optimized version of the kernel and expose it under the
# public name `kernel_syrk`, which is what the C driver calls.

kernel_syrk = rename(kernel_syrk_base, "kernel_syrk")

# 1. Hoist the common sub-expression alpha * A[i, k] into a scalar.
# This avoids recomputing it for every j and helps the compiler keep it in a register.
alpha_A_ik_exprs = kernel_syrk.find_all("alpha * A[i, k]")
kernel_syrk = bind_expr(kernel_syrk, alpha_A_ik_exprs, "aik")

# 2. Tile the reduction loop over k to improve cache locality for A and C.
#    We keep the (i, j) loop structure to preserve the lower-triangular access
#    pattern in C; only the k-dimension is blocked.
TK = 32  # tile size for the reduction dimension

k_loop = kernel_syrk.find_loop("for k in _:_")
kernel_syrk = divide_loop(kernel_syrk, k_loop, TK, ["ko", "ki"], tail="cut")

# 3. Unroll the inner k-tile loop to expose instruction-level parallelism
#    and enable better vectorization in the generated C.
ki_loop = kernel_syrk.find_loop("for ki in _:_")
kernel_syrk = unroll_loop(kernel_syrk, ki_loop)

# 4. Simplify the resulting code (constant folding, dead code elimination, etc.).
kernel_syrk = simplify(kernel_syrk)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_syrk (/*ctxt=*/NULL, n, m,
               (DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
               (DATA_TYPE*)POLYBENCH_ARRAY(C),
               (DATA_TYPE*)POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}