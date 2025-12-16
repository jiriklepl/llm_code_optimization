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
 *
 * Optimizations:
 * - Use row-major traversals for mean/stddev/normalization to match the
 *   row-major storage of `data`, improving spatial locality.
 * - Avoid repeated divisions: precompute 1/float_n and
 *   1/(sqrt(float_n) * stddev[j]) and replace per-element divisions
 *   with multiplications.
 * - Use `restrict`-qualified aliases for the main arrays to help the
 *   compiler with alias analysis and vectorization.
 * - Compute the correlation matrix with j-blocking (processing 8
 *   columns `j` at a time) so that the value of column `i` for a given
 *   row is reused across several dot products, reducing memory traffic.
 * - Still exploit symmetry: only the upper triangle is computed and
 *   mirrored to the lower triangle.
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

  /* Create local restrict-qualified aliases to help the compiler
     assume that these arrays do not alias one another. */
  DATA_TYPE (* restrict data_)[m]  = data;
  DATA_TYPE (* restrict corr_)[m]  = corr;
  DATA_TYPE * restrict mean_       = mean;
  DATA_TYPE * restrict stddev_     = stddev;

  const DATA_TYPE eps  = SCALAR_VAL(0.1);
  const DATA_TYPE one  = SCALAR_VAL(1.0);
  const DATA_TYPE zero = SCALAR_VAL(0.0);

  /* Precompute constants to reduce the number of divisions and calls
     to sqrt during the main loops. */
  const DATA_TYPE inv_float_n      = one / float_n;
  const DATA_TYPE sqrt_float_n     = SQRT_FUN(float_n);
  const DATA_TYPE inv_sqrt_float_n = one / sqrt_float_n;

  /* Per-column scaling factor used in normalization:
     inv_std_scale[j] = 1 / (sqrt(float_n) * stddev[j]) */
  DATA_TYPE inv_std_scale[m];

#pragma scop
  /* 1. Compute mean of each column:
        mean[j] = (1 / float_n) * sum_i data[i][j].

     Implementation detail:
     We traverse `data` row-wise to match its row-major layout, which
     significantly improves cache usage compared to a pure column-wise
     traversal.
  */
  for (j = 0; j < _PB_M; ++j)
    mean_[j] = zero;

  for (i = 0; i < _PB_N; ++i)
    {
      DATA_TYPE * restrict row = data_[i];
      for (j = 0; j < _PB_M; ++j)
        mean_[j] += row[j];
    }

  for (j = 0; j < _PB_M; ++j)
    mean_[j] *= inv_float_n;

  /* 2. Compute standard deviation of each column:
        stddev[j] = sqrt( (1 / float_n) *
                          sum_i (data[i][j] - mean[j])^2 )

     Again, we use a row-major traversal for better locality and
     compute (data - mean) once per element.
  */
  for (j = 0; j < _PB_M; ++j)
    stddev_[j] = zero;

  for (i = 0; i < _PB_N; ++i)
    {
      DATA_TYPE * restrict row = data_[i];
      for (j = 0; j < _PB_M; ++j)
        {
          DATA_TYPE diff = row[j] - mean_[j];
          stddev_[j] += diff * diff;
        }
    }

  for (j = 0; j < _PB_M; ++j)
    {
      stddev_[j] *= inv_float_n;
      stddev_[j] = SQRT_FUN(stddev_[j]);
      /* Handle near-zero std. dev. to avoid division by zero later. */
      stddev_[j] = stddev_[j] <= eps ? one : stddev_[j];
    }

  /* 3. Precompute normalization coefficients:
        inv_std_scale[j] = 1 / (sqrt(float_n) * stddev[j])

     This lets us replace a division per matrix element by a single
     multiplication in the normalization step.
  */
  for (j = 0; j < _PB_M; ++j)
    inv_std_scale[j] = inv_sqrt_float_n / stddev_[j];

  /* 4. Center and normalize the column vectors (z-score):
        data[i][j] <- (data[i][j] - mean[j]) /
                       (sqrt(float_n) * stddev[j])

     Implemented as:
        data[i][j] = (data[i][j] - mean[j]) * inv_std_scale[j];
  */
  for (i = 0; i < _PB_N; ++i)
    {
      DATA_TYPE * restrict row = data_[i];
      for (j = 0; j < _PB_M; ++j)
        row[j] = (row[j] - mean_[j]) * inv_std_scale[j];
    }

  /* 5. Calculate the m * m correlation matrix.

       After the normalization above, the correlation matrix is:
         corr = data^T * data

       We exploit symmetry (corr[i][j] == corr[j][i]) and compute only
       the upper triangle, mirroring to the lower triangle. To improve
       data reuse, we process the dot products in blocks of 8 columns
       of `j` for a fixed `i`, so that the value of the i-th column
       for row k is loaded once and reused across eight dot products.
  */
  for (i = 0; i < _PB_M; ++i)
    {
      /* Diagonal elements are exactly 1.0 by construction. */
      corr_[i][i] = one;

      const int J_BLOCK = 8;
      int j_block;

      /* Blocked computation over j: handle full blocks of 8 columns. */
      for (j_block = i + 1; j_block + J_BLOCK <= _PB_M; j_block += J_BLOCK)
        {
          DATA_TYPE sum0 = zero, sum1 = zero, sum2 = zero, sum3 = zero;
          DATA_TYPE sum4 = zero, sum5 = zero, sum6 = zero, sum7 = zero;

          for (k = 0; k < _PB_N; ++k)
            {
              DATA_TYPE * restrict row = data_[k];
              DATA_TYPE a = row[i];

              /* Compute 8 dot products in parallel for this row. */
              sum0 += a * row[j_block    ];
              sum1 += a * row[j_block + 1];
              sum2 += a * row[j_block + 2];
              sum3 += a * row[j_block + 3];
              sum4 += a * row[j_block + 4];
              sum5 += a * row[j_block + 5];
              sum6 += a * row[j_block + 6];
              sum7 += a * row[j_block + 7];
            }

          /* Store the results and exploit symmetry. */
          corr_[i][j_block    ] = sum0; corr_[j_block    ][i] = sum0;
          corr_[i][j_block + 1] = sum1; corr_[j_block + 1][i] = sum1;
          corr_[i][j_block + 2] = sum2; corr_[j_block + 2][i] = sum2;
          corr_[i][j_block + 3] = sum3; corr_[j_block + 3][i] = sum3;
          corr_[i][j_block + 4] = sum4; corr_[j_block + 4][i] = sum4;
          corr_[i][j_block + 5] = sum5; corr_[j_block + 5][i] = sum5;
          corr_[i][j_block + 6] = sum6; corr_[j_block + 6][i] = sum6;
          corr_[i][j_block + 7] = sum7; corr_[j_block + 7][i] = sum7;
        }

      /* Handle remaining columns that do not form a full block. */
      for (j = j_block; j < _PB_M; ++j)
        {
          if (j <= i)
            continue; /* Only upper triangle; diagonal already set. */

          DATA_TYPE sum = zero;
          for (k = 0; k < _PB_N; ++k)
            {
              DATA_TYPE * restrict row = data_[k];
              sum += row[i] * row[j];
            }

          corr_[i][j] = sum;
          corr_[j][i] = sum;
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