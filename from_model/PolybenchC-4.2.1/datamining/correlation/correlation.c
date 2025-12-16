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

   Optimizations vs. the original version:
   - All sweeps over the data matrix use row-major order (row i outer, column j
     inner) to match the C layout and improve cache locality.
   - Mean computation is split into initialization, accumulation, and scaling,
     with the reciprocal 1/float_n computed once and reused.
   - Variance accumulation and centering are fused: we compute
         diff = data[i][j] - mean[j]
       once, add diff*diff to the variance accumulator, and overwrite data with
       diff. This reduces memory traffic by one full pass over the matrix.
   - Normalization uses precomputed per-column scaling factors
         scale[j] = 1 / (sqrt(float_n) * stddev[j])
     so the hot N*M loop performs only a multiply per element instead of a
     division and a sqrt.
   - The correlation matrix is computed as a symmetric rank‑k update:
         corr += data^T * data
     over the upper triangle, using a loop order that walks both data and corr
     mostly in row-major order.
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

  /* Small constant to avoid division by very small stddev values. */
  const DATA_TYPE eps = SCALAR_VAL(0.1);

  /* Per-column scaling factors:
       scale[j] = 1 / (sqrt(float_n) * stddev[j])
     This extra temporary is O(M) and keeps the normalization loop free
     of divisions and extra sqrt calls. */
  DATA_TYPE scale[M];

#pragma scop
  /* --------------------------------------------------------------
   * 1. Compute column means:
   *      mean[j] = (1 / float_n) * sum_i data[i][j]
   *    We traverse the data matrix in row-major order: i outer, j inner.
   * -------------------------------------------------------------- */

  /* Zero-initialize means. */
  for (j = 0; j < _PB_M; j++)
    mean[j] = SCALAR_VAL(0.0);

  /* Accumulate sums over rows. */
  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE *data_i = data[i]; /* pointer to current row for faster access */
      for (j = 0; j < _PB_M; j++)
        mean[j] += data_i[j];
    }

  /* Precompute 1/float_n once and scale the column sums. */
  const DATA_TYPE inv_float_n = SCALAR_VAL(1.0) / float_n;
  for (j = 0; j < _PB_M; j++)
    mean[j] *= inv_float_n;

  /* --------------------------------------------------------------
   * 2. Compute column variances and center the data in-place.
   *
   *    After this phase:
   *      - data[i][j] holds (original_data[i][j] - mean[j])
   *      - stddev[j] holds sum_i (data[i][j])^2  (unnormalized variance)
   * -------------------------------------------------------------- */

  for (j = 0; j < _PB_M; j++)
    stddev[j] = SCALAR_VAL(0.0);

  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE *data_i = data[i];
      for (j = 0; j < _PB_M; j++)
        {
          DATA_TYPE diff = data_i[j] - mean[j];
          data_i[j] = diff;
          stddev[j] += diff * diff;
        }
    }

  /* Finish stddev computation: divide by N, take sqrt, and apply epsilon clamp. */
  for (j = 0; j < _PB_M; j++)
    {
      DATA_TYPE variance = stddev[j] * inv_float_n;
      DATA_TYPE std = SQRT_FUN(variance);
      /* The following is the usual way to handle near-zero std. dev. values,
         which below would cause a division by (almost) zero. */
      std = std <= eps ? SCALAR_VAL(1.0) : std;
      stddev[j] = std;
    }

  /* --------------------------------------------------------------
   * 3. Precompute per-column scaling factors:
   *      scale[j] = 1 / (sqrt(float_n) * stddev[j])
   *    This hoists SQRT_FUN(float_n) and the division by stddev[j]
   *    out of the hot N*M normalization loop.
   * -------------------------------------------------------------- */
  const DATA_TYPE sqrt_float_n      = SQRT_FUN(float_n);
  const DATA_TYPE inv_sqrt_float_n  = SCALAR_VAL(1.0) / sqrt_float_n;

  for (j = 0; j < _PB_M; j++)
    scale[j] = inv_sqrt_float_n / stddev[j];

  /* --------------------------------------------------------------
   * 4. Normalize data in-place.
   *
   *    At this point data[i][j] == original_data[i][j] - mean[j];
   *    we now scale it so that each column has unit 2-norm:
   *
   *      data[i][j] <- data[i][j] * scale[j]
   *
   *    which is algebraically equivalent to the original:
   *
   *      (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
   * -------------------------------------------------------------- */
  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE *data_i = data[i];
      for (j = 0; j < _PB_M; j++)
        data_i[j] *= scale[j];
    }

  /* --------------------------------------------------------------
   * 5. Compute the M x M correlation matrix on normalized data.
   *
   *    After normalization, the correlation between variables p and q is:
   *
   *      corr[p][q] = sum_k data[k][p] * data[k][q]
   *
   *    We compute only the upper triangle (including the diagonal) and then
   *    symmetrize. The diagonal is set explicitly to 1.0 to match the
   *    original implementation.
   * -------------------------------------------------------------- */

  /* Initialize upper triangle (including diagonal) to zero. */
  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      corr[i][j] = SCALAR_VAL(0.0);

  /* Symmetric rank‑k update: corr += data^T * data, upper triangle only.
     Loop order (k outer, then i, then j) walks both data and corr mostly
     in row-major order for good spatial locality. */
  for (k = 0; k < _PB_N; k++)
    {
      DATA_TYPE *data_k = data[k];
      for (i = 0; i < _PB_M; i++)
        {
          DATA_TYPE x = data_k[i];
          for (j = i; j < _PB_M; j++)
            corr[i][j] += x * data_k[j];
        }
    }

  /* Enforce symmetry and set the diagonal exactly to 1.0. */
  for (i = 0; i < _PB_M; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i + 1; j < _PB_M; j++)
        corr[j][i] = corr[i][j];
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