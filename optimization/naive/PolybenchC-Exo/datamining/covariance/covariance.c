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

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Baseline implementation with improved arithmetic and memory access patterns.
# Further optimized and parallelized below via scheduling primitives.
@proc
def kernel_covariance_impl(
    m: size,
    n: size,
    float_n: DATA_TYPE,
    data: DATA_TYPE[n, m] @ DRAM,
    cov: DATA_TYPE[m, m] @ DRAM,
    mean: DATA_TYPE[m] @ DRAM,
):

    # Optional safety assertion to avoid division by zero when n == 1.
    assert n > 1

    # Precompute reciprocals to avoid repeated divisions in inner loops.
    inv_float_n: DATA_TYPE
    inv_float_n = 1.0 / float_n

    inv_n_minus_1: DATA_TYPE
    inv_n_minus_1 = 1.0 / (float_n - 1.0)

    # 1. Compute column means.
    # 1a. Initialize means to zero.
    for j in seq(0, m):
        mean[j] = 0.0

    # 1b. Accumulate sum of each column using row-major access over data.
    #     This improves spatial locality compared to looping over columns first.
    for i in seq(0, n):
        for j in seq(0, m):
            mean[j] += data[i, j]

    # 1c. Normalize by the number of samples.
    for j in seq(0, m):
        mean[j] = mean[j] * inv_float_n

    # 2. Center the data: subtract the mean from each column.
    #    This loop is parallelized by the schedule below.
    for i in seq(0, n):
        for j in seq(0, m):
            data[i, j] = data[i, j] - mean[j]

    # 3. Compute the covariance matrix.
    #    We form each row i of the upper triangle using efficient row-wise access.
    #    For each feature index i, we:
    #      - zero cov[i, j] for j >= i
    #      - accumulate outer products over all samples k
    #      - scale by 1 / (n - 1)
    #      - mirror cov[i, j] into cov[j, i]
    for i in seq(0, m):

        # 3a. Zero the upper-triangular part of row i (including diagonal).
        for j in seq(i, m):
            cov[i, j] = 0.0

        # 3b. Accumulate outer products for the centered data.
        #     For each row k, we reuse x_ik across all j >= i,
        #     reading data[k, j] in a row-major (contiguous) fashion.
        for k in seq(0, n):
            x_ik: DATA_TYPE
            x_ik = data[k, i]
            for j in seq(i, m):
                cov[i, j] += x_ik * data[k, j]

        # 3c. Scale by 1 / (n - 1) and mirror to the lower triangle.
        for j in seq(i, m):
            cov[i, j] = cov[i, j] * inv_n_minus_1
            cov[j, i] = cov[i, j]


# Create the scheduled, optimized kernel that will be compiled to C.
kernel_covariance = rename(kernel_covariance_impl, "kernel_covariance")

# Parallelize the data-centering loop over rows.
# This loop writes distinct rows of 'data' and only reads 'mean', so it is safe.
center_i_loop = kernel_covariance.find_loop("i #1")
kernel_covariance = parallelize_loop(kernel_covariance, center_i_loop)

# Parallelize the outer covariance loop over feature dimension i.
# Each iteration of this loop touches a disjoint set of cov[i, j] / cov[j, i],
# so different iterations can safely run in parallel.
cov_i_loop = kernel_covariance.find_loop("i #2")
kernel_covariance = parallelize_loop(kernel_covariance, cov_i_loop)
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