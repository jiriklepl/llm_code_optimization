/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


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


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Column means are computed using row-major traversal to improve cache use.
   - Divisions by `float_n` and `float_n - 1` are turned into multiplications
     by precomputed reciprocals.
   - The covariance is computed via an outer-product accumulation that
     traverses rows of `data` (row-major friendly) and writes contiguous
     segments of `cov`.
   - Local `__restrict` aliases are introduced to help the compiler
     with alias analysis and vectorization.
*/
static
void kernel_covariance[[gnu::flatten, gnu::noinline]](int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  /* Local restricted pointers to help the compiler optimize memory accesses.
     Types are identical to the original parameters; only the `restrict`
     qualifier is added on the pointers themselves. */
  DATA_TYPE (* __restrict data_)[m] = data;
  DATA_TYPE (* __restrict cov_)[m]  = cov;
  DATA_TYPE * __restrict mean_      = mean;

#pragma scop
  /* 1. Compute column means.
     Original version:
       for (j) { mean[j]=0; for (i) mean[j]+=data[i][j]; mean[j]/=float_n; }
     New version:
       - Initialize all means to 0.
       - Accumulate using row-major traversal (i outer, j inner).
       - Scale by 1/float_n afterwards.
     The numerical sequence for each mean[j] is still:
       0 + data[0][j] + ... + data[N-1][j], then one division. */

  for (j = 0; j < _PB_M; j++)
    mean_[j] = SCALAR_VAL(0.0);

  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE * __restrict data_row = data_[i];
    for (j = 0; j < _PB_M; j++)
      mean_[j] += data_row[j];
  }

  const DATA_TYPE inv_float_n = SCALAR_VAL(1.0) / float_n;

  for (j = 0; j < _PB_M; j++)
    mean_[j] *= inv_float_n;


  /* 2. Center the data: subtract the mean of each column.
     Traversal is row-major (i outer, j inner) to match storage. */
  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE * __restrict data_row = data_[i];
    for (j = 0; j < _PB_M; j++)
      data_row[j] -= mean_[j];
  }


  /* 3. Compute covariance matrix.
     Original algorithm:
       for (i)
         for (j = i .. M-1) {
           cov[i][j] = 0;
           for (k)
             cov[i][j] += data[k][i] * data[k][j];
           cov[i][j] /= (float_n - 1);
           cov[j][i] = cov[i][j];
         }

     New algorithm (outer-product formulation):
       - Initialize cov[i][j] = 0 for i <= j (upper triangle).
       - For each row k, accumulate:
           cov[i][j] += data[k][i] * data[k][j]   for all i <= j
         using row-major access to `data[k][*]`.
       - After accumulation, scale cov[i][j] by 1/(float_n - 1)
         and explicitly set cov[j][i] = cov[i][j].

     For each (i,j), the sequence of operations on cov[i][j] is still:
       cov[i][j] = 0;
       for k = 0..N-1: cov[i][j] += data[k][i]*data[k][j];
       cov[i][j] *= 1/(float_n - 1);
       cov[j][i] = cov[i][j];
     Only the interleaving between different (i,j) pairs changes,
     which does not affect the value of each element. */

  /* Initialize upper triangle of covariance to zero. */
  for (i = 0; i < _PB_M; i++)
  {
    DATA_TYPE * __restrict cov_row_i = cov_[i];
    for (j = i; j < _PB_M; j++)
      cov_row_i[j] = SCALAR_VAL(0.0);
  }

  /* Accumulate outer products row by row. */
  for (k = 0; k < _PB_N; k++)
  {
    DATA_TYPE * __restrict row = data_[k];

    for (i = 0; i < _PB_M; i++)
    {
      const DATA_TYPE row_i = row[i];
      DATA_TYPE * __restrict cov_row_i = cov_[i];

      /* Only upper triangle; j runs over a contiguous segment of cov_row_i. */
      for (j = i; j < _PB_M; j++)
        cov_row_i[j] += row_i * row[j];
    }
  }

  const DATA_TYPE inv_denom =
    SCALAR_VAL(1.0) / (float_n - SCALAR_VAL(1.0));

  /* Scale and symmetrize covariance matrix. */
  for (i = 0; i < _PB_M; i++)
  {
    DATA_TYPE * __restrict cov_row_i = cov_[i];
    for (j = i; j < _PB_M; j++)
    {
      cov_row_i[j] *= inv_denom;
      cov_[j][i] = cov_row_i[j];
    }
  }
#pragma endscop

}


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

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

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