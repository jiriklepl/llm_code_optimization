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

# Tunable block sizes. These are compile-time constants in the Exo world.
# TILE_I: block rows of C / A to improve cache locality and enable coarse-grain parallelism.
# TILE_K: block the reduction dimension m to increase reuse of A and keep working sets in cache.
TILE_I = 32
TILE_K = 64


@proc
def kernel_syrk(
    n: size,
    m: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[n, n] @ DRAM,
    A: DATA_TYPE[n, m] @ DRAM,
):
    # Basic sanity: sizes must be positive for the tiling formulas to make sense.
    assert n > 0
    assert m > 0

    # Temporary buffer for one output row of the lower triangle.
    # For a fixed row i, we accumulate:
    #   tmp[j] = sum_k A[i, k] * A[j, k]   for 0 <= j <= i
    # and then apply the final alpha scaling once per entry:
    #   C[i, j] = beta * C[i, j] + alpha * tmp[j]
    #
    # This reduces the number of multiplications by alpha from O(n^2 * m)
    # (original kernel) to O(n^2), and allows us to keep the partial sums
    # in a contiguous buffer, minimizing traffic to C.
    tmp: DATA_TYPE[n]

    # Block outer loop over rows of C/A to improve locality.
    # We iterate tiles of size TILE_I; each tile processes several rows.
    for I in seq(0, (n + TILE_I - 1) / TILE_I):
        for i in seq(0, TILE_I):
            ii: size
            ii = I * TILE_I + i

            # Guard for the last (possibly partial) tile.
            if ii < n:
                # ------------------------------------------------------------------
                # 1. Scale the lower-triangular part of row ii by beta:
                #    C[ii, j] = beta * C[ii, j],  for 0 <= j <= ii
                # ------------------------------------------------------------------
                for j in seq(0, ii + 1):
                    C[ii, j] = beta * C[ii, j]

                # ------------------------------------------------------------------
                # 2. Initialize accumulators for this row:
                #    tmp[j] = 0, for 0 <= j <= ii
                # ------------------------------------------------------------------
                for j in seq(0, ii + 1):
                    tmp[j] = 0.0

                # ------------------------------------------------------------------
                # 3. Blocked rank-k update:
                #    tmp[j] += A[ii, k] * A[j, k]
                #    over all k in [0, m), in tiles of size TILE_K.
                #    We only touch the lower triangle (j <= ii).
                # ------------------------------------------------------------------
                for K in seq(0, (m + TILE_K - 1) / TILE_K):
                    for k in seq(0, TILE_K):
                        kk: size
                        kk = K * TILE_K + k

                        if kk < m:
                            aik: DATA_TYPE
                            aik = A[ii, kk]

                            # j is the inner loop: C[ii, j] and A[j, kk] are
                            # both row-major in j, so these accesses are
                            # contiguous along the row dimension.
                            for j in seq(0, ii + 1):
                                tmp[j] += aik * A[j, kk]

                # ------------------------------------------------------------------
                # 4. Final update:
                #    C[ii, j] += alpha * tmp[j],  for 0 <= j <= ii
                #    This implements:
                #      C[ii, j] = beta * C[ii, j] + alpha * sum_k A[ii, k] * A[j, k]
                #    exactly for the lower triangle, matching the original kernel.
                # ------------------------------------------------------------------
                for j in seq(0, ii + 1):
                    C[ii, j] += alpha * tmp[j]
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