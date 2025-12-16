/**
 * Exo covariance driver: mirrors PolyBench/C covariance.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"

/* Include the Exo-generated kernel header. */
#include "generated/covariance/covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}

/*
 * Optimized kernel in Exo:
 * - Means are accumulated in a row-major traversal of `data`
 *   to match its layout and improve cache use.
 * - The covariance is formed as a sequence of outer products
 *   of centered rows:
 *       cov = (1/(n-1)) * sum_r x_r^T * x_r
 *   where x_r is the centered row r.
 *   This is algebraically equivalent to the original triple
 *   loop over (i, j, k) but has better memory locality:
 *     - inner loops walk contiguous feature indices,
 *     - each centered element data[r, c] is reused from cache
 *       when contributing to multiple cov[i, j] pairs.
 * - Divisions by float_n and (float_n - 1) inside loops are
 *   replaced by multiplies with precomputed reciprocals.
 */

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

@proc
def kernel_covariance(
    m: size,
    n: size,
    float_n: DATA_TYPE,
    data: DATA_TYPE[n, m] @ DRAM,
    cov: DATA_TYPE[m, m] @ DRAM,
    mean: DATA_TYPE[m] @ DRAM,
):
    # Precompute reciprocals to avoid repeated divisions in the loops.
    inv_float_n: DATA_TYPE
    inv_float_n_minus1: DATA_TYPE

    inv_float_n = 1.0 / float_n
    inv_float_n_minus1 = 1.0 / (float_n - 1.0)

    # ------------------------------------------------------------
    # 1. Compute the mean of each column j over all rows i.
    #    We zero the means and then accumulate in a row-major
    #    traversal of `data` to match its storage layout.
    # ------------------------------------------------------------
    for j in seq(0, m):
        mean[j] = 0.0

    for i in seq(0, n):
        for j in seq(0, m):
            mean[j] += data[i, j]

    for j in seq(0, m):
        mean[j] = mean[j] * inv_float_n

    # ------------------------------------------------------------
    # 2. Center the data in-place:
    #       data[i, j] <- data[i, j] - mean[j]
    #    This pass is already row-major and cache friendly.
    # ------------------------------------------------------------
    for i in seq(0, n):
        for j in seq(0, m):
            data[i, j] = data[i, j] - mean[j]

    # ------------------------------------------------------------
    # 3. Initialize the upper triangle of the covariance matrix.
    #    We only touch entries with j >= i; the lower triangle
    #    will be filled by symmetry later.
    # ------------------------------------------------------------
    for i_cov in seq(0, m):
        for j_cov in seq(i_cov, m):
            cov[i_cov, j_cov] = 0.0

    # ------------------------------------------------------------
    # 4. Accumulate covariance via outer products of centered rows.
    #
    # For each centered row r:
    #   for i_cov:
    #     x_i = data[r, i_cov]
    #     for j_cov in [i_cov, m):
    #       cov[i_cov, j_cov] += x_i * data[r, j_cov]
    #
    # This computes:
    #   cov[i, j] = sum_{r=0}^{n-1} data[r, i] * data[r, j]
    # for all j >= i, which matches the original covariance
    # definition but accesses `data` and `cov` in a way that
    # respects row-major layout.
    # ------------------------------------------------------------
    x_i: DATA_TYPE

    for r in seq(0, n):
        for i_cov in seq(0, m):
            x_i = data[r, i_cov]
            for j_cov in seq(i_cov, m):
                cov[i_cov, j_cov] += x_i * data[r, j_cov]

    # ------------------------------------------------------------
    # 5. Normalize by 1 / (n - 1) and enforce symmetry explicitly.
    #    This matches the original kernel, which divides each
    #    cov[i, j] by (float_n - 1) and then sets cov[j, i] =
    #    cov[i, j].
    # ------------------------------------------------------------
    for i_cov in seq(0, m):
        for j_cov in seq(i_cov, m):
            cov[i_cov, j_cov] = cov[i_cov, j_cov] * inv_float_n_minus1
            cov[j_cov, i_cov] = cov[i_cov, j_cov]
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_covariance (/*ctxt=*/NULL, m, n,
                     (DATA_TYPE*)&float_n,
                     (DATA_TYPE*)POLYBENCH_ARRAY(data),
                     (DATA_TYPE*)POLYBENCH_ARRAY(cov),
                     (DATA_TYPE*)POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}