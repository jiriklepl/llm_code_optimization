/**
 * Exo TRMM driver: mirrors PolyBench/C trmm.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"

/* Include the Exo-generated kernel header. */
#include "generated/trmm/trmm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

# Core Exo imports.
from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# ---------------------------------------------------------------------------
# Baseline triangular matrix-matrix multiply kernel.
# This implements the same computation as the original PolyBench TRMM kernel:
#
#   for i in 0..m-1
#     for j in 0..n-1
#       for k in i+1..m-1
#         B[i,j] += A[k,i] * B[k,j]
#       B[i,j] = alpha * B[i,j]
#
# A is unit lower-triangular (diagonal = 1, strictly upper part unused),
# and B is an m x n dense matrix.
# ---------------------------------------------------------------------------

@proc
def kernel_trmm_unopt(
    m: size,
    n: size,
    alpha: DATA_TYPE,
    A: DATA_TYPE[m, m] @ DRAM,
    B: DATA_TYPE[m, n] @ DRAM,
):
    for i in seq(0, m):
        for j in seq(0, n):
            for k in seq(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]


# ---------------------------------------------------------------------------
# Optimization schedule
#
# Goals:
#  - Preserve the original row-major access pattern for B[i, j].
#  - Improve cache locality and enable better vectorization/parallelism
#    along the contiguous j dimension.
#  - Keep the triangular data-dependence in the i / k dimensions intact.
# ---------------------------------------------------------------------------

# 1. Rename to the public entry-point that the C driver calls.
kernel_trmm = rename(kernel_trmm_unopt, "kernel_trmm")

# 2. Tile the j loop.
#
# We split:
#     for j in seq(0, n):
#       ...
# into:
#     for jo in seq(0, ceil(n / tile_j)):
#       for ji in seq(0, tile_j):
#         j = jo * tile_j + ji
#         if j < n:
#             ...
#
# Tiling along j keeps accesses to B[i, j] and B[k, j] contiguous within
# each tile, improving spatial locality and making the inner loop body
# more amenable to auto-vectorization by the C compiler.
tile_j = 32
kernel_trmm = divide_loop(kernel_trmm, "j", tile_j, ["jo", "ji"], tail="guard")

# 3. Parallelize across j-tiles.
#
# For a fixed row i, different tiles (different jo values) operate on
# disjoint column ranges of B and only read from A and other rows of B.
# There are no cross-iteration write-write or write-read dependences
# along jo, so it is safe to run these tiles in parallel.
#
# This lowers to an OpenMP 'parallel for' pragma in the generated C code.
kernel_trmm = parallelize_loop(kernel_trmm, "jo")

# 4. Simplify the resulting code to clean up index arithmetic and guards.
kernel_trmm = simplify(kernel_trmm)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_trmm (/*ctxt=*/NULL, m, n,
               (DATA_TYPE*)&alpha,
               (DATA_TYPE*)POLYBENCH_ARRAY(A),
               (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}