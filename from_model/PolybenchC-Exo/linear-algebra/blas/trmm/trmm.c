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

from exo import *
from exo.API_scheduling import *  # import scheduling primitives (not used directly here but available)
from exo.libs.memories import DRAM

# Tile size for the column dimension (j) of B.
# Chosen to keep panels of B[i, :] and B[k, :] in cache.
TILE_J = 64

@proc
def kernel_trmm(
    m: size,
    n: size,
    alpha: DATA_TYPE,
    A: DATA_TYPE[m, m] @ DRAM,
    B: DATA_TYPE[m, n] @ DRAM,
):
    """
    Computes in-place:
        B := alpha * A^T * B

    where A is unit lower-triangular (only A[k, i] for k > i are used)
    and both A and B are stored row-major, matching the C driver.

    Compared to the original i–j–k kernel:

        for i in 0..m-1:
            for j in 0..n-1:
                for k in i+1..m-1:
                    B[i, j] += A[k, i] * B[k, j]
                B[i, j] = alpha * B[i, j]

    this implementation uses an outer-product style triangular update:

        for k in 0..m-1:
            for i in 0..k-1:
                for j in 0..n-1:
                    B[i, j] += A[k, i] * B[k, j]

    followed by a separate scaling pass:

        for i, j:
            B[i, j] *= alpha

    This reordering:
      * preserves the exact mathematical result
        B[i, j] = alpha * ( B0[i, j] + sum_{k>i} A[k, i] * B0[k, j] )
      * reads A[k, i] along rows (contiguous in memory),
      * and streams B[i, j] and B[k, j] contiguously in j, improving
        cache behavior and vectorization opportunities.
    """

    # Stage 1: triangular update in k–i–j order with j-blocking.
    for k in seq(0, m):
        # Only rows strictly above k (i < k) are updated from row k.
        for i in seq(0, k):
            # Reuse A[k, i] across the whole update of row i from row k.
            a_ki: DATA_TYPE
            a_ki = A[k, i]

            # Block the column dimension of B to improve cache locality.
            # Each tile J covers a contiguous panel of columns.
            for J in seq(0, (n + TILE_J - 1) / TILE_J):
                for j in seq(0, TILE_J):
                    jj = TILE_J * J + j
                    if jj < n:
                        # B[i, jj] and B[k, jj] are both row-major and
                        # contiguous in jj, so this inner loop is
                        # cache- and SIMD-friendly.
                        B[i, jj] += a_ki * B[k, jj]

    # Stage 2: scale the completed result by alpha.
    # Because the triangular update is linear in B, scaling once at the end
    # is equivalent to scaling inside the original kernel.
    for i in seq(0, m):
        for J in seq(0, (n + TILE_J - 1) / TILE_J):
            for j in seq(0, TILE_J):
                jj = TILE_J * J + j
                if jj < n:
                    B[i, jj] = alpha * B[i, jj]
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