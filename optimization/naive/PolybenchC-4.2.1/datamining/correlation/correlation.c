/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use local restrict-qualified aliases for better auto-vectorization.
   - Traverse `data` row-wise (i-major) for mean/stddev to improve
     spatial locality while preserving per-column summation order.
   - Hoist SQRT_FUN(float_n) out of inner loops and precompute
     per-column scaling factors to avoid redundant work in the
     normalization phase.
   - Reformulate the correlation computation as an outer-product
     accumulation (k-major) to access `data` row-wise and to work
     on contiguous segments of `corr`, improving cache use and
     vectorization, while preserving the accumulation order for
     each corr[i][j].
*/
static
void kernel_correlation[[gnu::flatten, gnu::noinline]](int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);

  /* Local restrict-qualified aliases to help the compiler reason about
     aliasing and enable more aggressive vectorization.  The PolyBench
     allocation scheme guarantees that these arrays do not overlap. */
  DATA_TYPE (*restrict data_)[m]   = data;
  DATA_TYPE (*restrict corr_)[m]   = corr;
  DATA_TYPE *restrict mean_        = mean;
  DATA_TYPE *restrict stddev_      = stddev;

#pragma scop
  /* 1. Compute column means.
     We first zero-initialize all means, then accumulate them in a
     row-major traversal of `data`.  For each column j, the sequence
     of additions (over i = 0.._PB_N-1) is identical to the original
     code, so floating-point reduction order per column is preserved. */
  for (j = 0; j < _PB_M; j++)
    mean_[j] = SCALAR_VAL(0.0);

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      mean_[j] += data_[i][j];

  for (j = 0; j < _PB_M; j++)
    mean_[j] /= float_n;

  /* 2. Compute standard deviations.
     Same idea: zero-initialize, then traverse `data` row-wise while
     preserving, for each column j, the order of accumulation over i. */
  for (j = 0; j < _PB_M; j++)
    stddev_[j] = SCALAR_VAL(0.0);

  for (i = 0; i < _PB_N; i++)
    {
      for (j = 0; j < _PB_M; j++)
        {
          DATA_TYPE diff = data_[i][j] - mean_[j];
          stddev_[j] += diff * diff;
        }
    }

  for (j = 0; j < _PB_M; j++)
    {
      stddev_[j] /= float_n;
      stddev_[j] = SQRT_FUN(stddev_[j]);
      /* Handle near-zero std. dev. values to avoid division by zero. */
      stddev_[j] = stddev_[j] <= eps ? SCALAR_VAL(1.0) : stddev_[j];
    }

  /* 3. Center and reduce the column vectors.
     Hoist SQRT_FUN(float_n) out of the inner loops and precompute
     per-column denominators sqrt(float_n) * stddev[j].  For each
     column j, the denominator is computed once and reused; since its
     operands are identical to those in the original expression,
     the value is bit-identical for that column. */
  DATA_TYPE sqrt_n = SQRT_FUN(float_n);

  /* Per-column normalization factors: denom[j] = sqrt_n * stddev[j]. */
  DATA_TYPE denom[_PB_M];
  for (j = 0; j < _PB_M; j++)
    denom[j] = sqrt_n * stddev_[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data_[i][j] -= mean_[j];
        data_[i][j] /= denom[j];
      }

  /* 4. Calculate the m * m correlation matrix.

     We compute the upper triangle (i < j) using an outer-product style
     accumulation over rows k:

       corr[i][j] = sum_k data[k][i] * data[k][j],  i < j

     - First, initialize the strict upper triangle to zero.
     - Then, for each row k, update all corr[i][j], i < j.
     - Finally, set the diagonal to 1.0 and mirror the upper triangle
       into the lower triangle.

     For each pair (i, j) with i < j, the contributions are added in
     increasing k order (0.._PB_N-1), exactly as in the original
     code, so the floating-point reduction order per entry is preserved.
  */

  /* Initialize strict upper triangle to zero. */
  for (i = 0; i < _PB_M; i++)
    for (j = i + 1; j < _PB_M; j++)
      corr_[i][j] = SCALAR_VAL(0.0);

  /* Outer-product style accumulation: k-major, then i, then j.
     This accesses `data` row-wise and, for fixed i, updates a
     contiguous segment corr_[i][i+1.._PB_M-1], which is cache-friendly
     and vectorizable. */
  for (k = 0; k < _PB_N; k++)
    {
      for (i = 0; i < _PB_M - 1; i++)
        {
          DATA_TYPE data_ki = data_[k][i];
          for (j = i + 1; j < _PB_M; j++)
            corr_[i][j] += data_ki * data_[k][j];
        }
    }

  /* Set diagonal elements to 1.0 and symmetrize the matrix. */
  for (i = 0; i < _PB_M; i++)
    {
      corr_[i][i] = SCALAR_VAL(1.0);
      for (j = i + 1; j < _PB_M; j++)
        corr_[j][i] = corr_[i][j];
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
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}