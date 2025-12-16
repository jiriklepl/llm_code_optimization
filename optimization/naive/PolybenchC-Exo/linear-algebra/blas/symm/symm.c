/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"

/* Include the Exo-generated kernel header. */
#include "generated/symm/symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
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

# Base kernel: direct transcription of the PolyBench SYMM kernel.
# C, A, and B are dense row-major matrices in DRAM.
@proc
def kernel_symm_base(
    m: size,
    n: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[m, n] @ DRAM,
    A: DATA_TYPE[m, m] @ DRAM,
    B: DATA_TYPE[m, n] @ DRAM,
):
    for i in seq(0, m):
        for j in seq(0, n):
            temp2: DATA_TYPE
            temp2 = 0
            for k in seq(0, i):
                # A is symmetric; only the lower triangle and diagonal are used.
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2


# --------------------
# Scheduling for kernel_symm_base
# --------------------

# 1) Common-subexpression elimination for alpha * B[i, j].
#    This value is invariant across the inner k-loop, so we compute it once
#    per (i, j) and reuse it.
s1 = bind_expr(
    kernel_symm_base,
    kernel_symm_base.find_all("alpha * B[i, j]"),
    "alpha_B_ij",
)

# 2) Tile the i loop to improve locality for rows of A and C.
I_TILE = 32
s2 = divide_loop(s1, "i", I_TILE, ("io", "ii"), tail="guard")

# 3) Tile the j loop so that inner tiles work on contiguous columns of B and C.
J_TILE = 32
s3 = divide_loop(s2, "j", J_TILE, ("jo", "ji"), tail="guard")

# 4) Unroll the inner j-tile loop (ji). Its trip count is the constant J_TILE,
#    and a guarded split ensures correctness for non-multiple sizes.
s4 = unroll_loop(s3, "ji")

# 5) Parallelize across j-tiles. Each jo-iteration touches disjoint columns
#    of C (and reads disjoint columns of B), so iterations are independent.
s5 = parallelize_loop(s4, "jo")

# 6) Simplify the resulting code to clean up algebra and dead code.
s6 = simplify(s5)

# 7) Export the scheduled kernel under the name expected by the C driver.
kernel_symm = rename(s6, "kernel_symm")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_symm (/*ctxt=*/NULL, m, n,
	       (DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
	       (DATA_TYPE*)POLYBENCH_ARRAY(C),
	       (DATA_TYPE*)POLYBENCH_ARRAY(A),
	       (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}