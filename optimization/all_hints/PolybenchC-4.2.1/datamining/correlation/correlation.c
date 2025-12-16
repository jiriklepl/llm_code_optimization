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

   Optimizations applied (semantics preserved):
   - Compute column means and variances in row-major order for better
     cache locality and auto-vectorization.
   - Precompute 1/float_n and 1/sqrt(float_n) to replace repeated
     divisions and square roots by cheaper multiplications.
   - Precompute per-column normalization factors to replace a division
     per element by a multiplication.
   - Reorganize correlation computation as a row-wise accumulation
     with j as the innermost loop for contiguous access to data and corr.
   - Parallelize the dominant correlation computation over the
     column index i using OpenMP; each thread owns disjoint entries
     of the upper triangle, so there are no write conflicts.

   All transformations are algebraically equivalent to the original
   computation; only the order of floating-point operations changes.
*/
static
void kernel_correlation [[gnu::flatten, gnu::noinline]] (int m, int n,
                     DATA_TYPE float_n,
                     DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
                     DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
                     DATA_TYPE POLYBENCH_1D(mean,M,m),
                     DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  /* Constants precomputed once to avoid redundant work inside loops. */
  const DATA_TYPE eps      = SCALAR_VAL(0.1);
  const DATA_TYPE one      = SCALAR_VAL(1.0);
  const DATA_TYPE zero     = SCALAR_VAL(0.0);
  const DATA_TYPE inv_n    = one / float_n;              /* 1 / N        */
  const DATA_TYPE inv_sqrt_n = one / SQRT_FUN(float_n);  /* 1 / sqrt(N)  */

  /* Per-column scaling factor:
     scale[j] = 1 / (sqrt(float_n) * stddev[j])
              = inv_sqrt_n / stddev[j].
     Allocated once on the stack; size is proportional to M and well
     below the requested memory overhead limit. */
  int mPB = _PB_M;
  DATA_TYPE scale[mPB];

#pragma scop
  /* ------------------------------------------------------------------
   * 1. Compute column means.
   *
   * Original:
   *   for (j) mean[j] = 0;
   *     for (i) mean[j] += data[i][j];
   *   mean[j] /= float_n;
   *
   * New version:
   *   - Initialize means to 0.
   *   - Accumulate per-row (row-major access: j innermost).
   *   - Scale by 1/N at the end.
   * ------------------------------------------------------------------ */

  /* Initialize means. */
  for (j = 0; j < _PB_M; j++)
    mean[j] = zero;

  /* Accumulate sums in row-major order for better locality. */
  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE *row = data[i];
    for (j = 0; j < _PB_M; j++)
      mean[j] += row[j];
  }

  /* Divide by N (using precomputed 1/N). */
  for (j = 0; j < _PB_M; j++)
    mean[j] *= inv_n;


  /* ------------------------------------------------------------------
   * 2. Compute column standard deviations.
   *
   * Original (per column j, scanning down rows):
   *   stddev[j] = 0;
   *   for (i) stddev[j] += (data[i][j] - mean[j])^2;
   *   stddev[j] /= float_n;
   *   stddev[j] = sqrt(stddev[j]);
   *   if (stddev[j] <= eps) stddev[j] = 1.0;
   *
   * New version:
   *   - Initialize stddev array to 0.
   *   - Accumulate squared differences in row-major order.
   *   - Post-process each column j as above.
   *   - Additionally compute scale[j] = 1/(sqrt(N)*stddev[j]).
   * ------------------------------------------------------------------ */

  /* Initialize variances. */
  for (j = 0; j < _PB_M; j++)
    stddev[j] = zero;

  /* Accumulate (x - mean)^2 per column, row-major. */
  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE *row = data[i];
    for (j = 0; j < _PB_M; j++)
    {
      DATA_TYPE diff = row[j] - mean[j];
      stddev[j] += diff * diff;
    }
  }

  /* Finalize standard deviations and compute per-column scale. */
  for (j = 0; j < _PB_M; j++)
  {
    stddev[j] *= inv_n;
    stddev[j] = SQRT_FUN(stddev[j]);

    /* Handle near-zero stddev to avoid division by ~0 later. */
    if (stddev[j] <= eps)
      stddev[j] = one;

    /* Precompute normalization factor:
       (x - mean[j]) / (sqrt(N) * stddev[j]) =
       (x - mean[j]) * (1 / (sqrt(N) * stddev[j])).
    */
    scale[j] = inv_sqrt_n / stddev[j];
  }


  /* ------------------------------------------------------------------
   * 3. Center and reduce the column vectors.
   *
   * Original:
   *   for (i)
   *     for (j) {
   *       data[i][j] -= mean[j];
   *       data[i][j] /= sqrt(float_n) * stddev[j];
   *     }
   *
   * New version:
   *   - Row-major order preserved.
   *   - Replace per-element division + sqrt by a multiply with scale[j].
   * ------------------------------------------------------------------ */

  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE *row = data[i];
    for (j = 0; j < _PB_M; j++)
    {
      row[j] = (row[j] - mean[j]) * scale[j];
    }
  }


  /* ------------------------------------------------------------------
   * 4. Compute the m x m correlation matrix.
   *
   * Original:
   *   for (i = 0; i < M-1; i++) {
   *     corr[i][i] = 1.0;
   *     for (j = i+1; j < M; j++) {
   *       corr[i][j] = 0.0;
   *       for (k = 0; k < N; k++)
   *         corr[i][j] += data[k][i] * data[k][j];
   *       corr[j][i] = corr[i][j];
   *     }
   *   }
   *   corr[M-1][M-1] = 1.0;
   *
   * This computes, for i < j:
   *   corr[i][j] = sum_k data[k][i] * data[k][j].
   *
   * New version:
   *   - First zero the upper triangular part (i < j).
   *   - Then, for each pair (i, j), accumulate contributions row by row.
   *     We keep j as innermost index to access both data and corr
   *     contiguously in memory.
   *   - Parallelize over i: each thread owns a distinct row i of the
   *     upper triangle, so no entries are updated by more than one thread.
   *   - Finally, set the diagonal to 1 and copy the upper triangle
   *     into the lower triangle.
   * ------------------------------------------------------------------ */

  /* Initialize upper triangle (excluding diagonal) to zero. */
  for (i = 0; i < _PB_M - 1; i++)
    for (j = i + 1; j < _PB_M; j++)
      corr[i][j] = zero;

  /* Accumulate the upper triangle.
     Each (i, j) with j > i is updated only in the i-th iteration,
     so "#pragma omp parallel for" is safe (no data races). */
#pragma omp parallel for private(j, k) schedule(static)
  for (i = 0; i < _PB_M - 1; i++)
  {
    for (k = 0; k < _PB_N; k++)
    {
      const DATA_TYPE *row = data[k];
      const DATA_TYPE x_i  = row[i];

      /* Inner loop has contiguous access in j dimension for both
         'row[j]' and 'corr[i][j]', which is cache- and SIMD-friendly. */
      for (j = i + 1; j < _PB_M; j++)
      {
        corr[i][j] += x_i * row[j];
      }
    }
  }

  /* Fill diagonal and lower triangle. */
  for (i = 0; i < _PB_M; i++)
  {
    corr[i][i] = one;
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