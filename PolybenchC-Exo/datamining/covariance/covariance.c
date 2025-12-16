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

@proc
def kernel_covariance(
    m: size,
    n: size,
    float_n: DATA_TYPE,
    data: DATA_TYPE[n, m] @ DRAM,
    cov: DATA_TYPE[m, m] @ DRAM,
    mean: DATA_TYPE[m] @ DRAM,
):
    # Compute the mean of each column j over all rows i
    for j in seq(0, m):
        mean[j] = 0.0
        for i in seq(0, n):
            mean[j] += data[i, j]
        mean[j] = mean[j] / float_n

    # Center the data by subtracting the mean from each column
    for i in seq(0, n):
        for j in seq(0, m):
            data[i, j] = data[i, j] - mean[j]

    # Compute the covariance matrix
    for i in seq(0, m):
        for j in seq(i, m):
            cov[i, j] = 0.0
            for k in seq(0, n):
                cov[i, j] += data[k, i] * data[k, j]
            cov[i, j] = cov[i, j] / (float_n - 1.0)
            cov[j, i] = cov[i, j]
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