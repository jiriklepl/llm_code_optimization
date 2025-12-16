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
   - Mean computation is reorganized to traverse `data` row‑major
     (i outer, j inner), improving spatial locality.
   - Mean division and covariance division are replaced with
     multiplications by precomputed reciprocals.
   - Covariance computation is rewritten as a symmetric
     rank‑1 update (outer‑product) over centered rows:
         cov += x_k^T * x_k
     which streams each data row once and reuses it fully.
   - The covariance update is limited to the upper triangle
     (j >= i), then mirrored to the lower triangle.
*/
static
void kernel_covariance[[gnu::flatten, gnu::noinline]](int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  /* Tile size along the feature (M) dimension for the covariance
     accumulation. This is a moderate value chosen to improve cache
     behavior and give the compiler a regular inner loop to vectorize.
     It does not need to divide M exactly; edge tiles are handled
     with min() logic. */
  const int tile_m = 32;

  /* Precompute reciprocals once; this removes O(M + M^2) scalar
     divisions from inner loops. */
  DATA_TYPE inv_float_n = SCALAR_VAL(1.0) / float_n;
  DATA_TYPE inv_float_n_minus_one =
      SCALAR_VAL(1.0) / (float_n - SCALAR_VAL(1.0));

#pragma scop

  /* ------------------------------------------------------------
   * 1. Compute per‑column means.
   *
   * Original:
   *   for (j)
   *     mean[j] = 0;
   *     for (i) mean[j] += data[i][j];
   *     mean[j] /= float_n;
   *
   * New version:
   *   - Separate initialization, accumulation, and scaling.
   *   - Accumulate in row‑major order (i outer, j inner) to
   *     stream `data` with unit‑stride accesses.
   *   - The mathematical result is the same:
   *       mean[j] = (1/N) * sum_i data[i][j].
   * ----------------------------------------------------------*/

  /* Initialize means to zero. */
  for (j = 0; j < _PB_M; j++)
    mean[j] = SCALAR_VAL(0.0);

  /* Accumulate column sums streaming `data` row‑major. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      mean[j] += data[i][j];

  /* Scale by 1/float_n (precomputed). */
  for (j = 0; j < _PB_M; j++)
    mean[j] *= inv_float_n;


  /* ------------------------------------------------------------
   * 2. Center the data matrix in place:
   *      data[i][j] <- data[i][j] - mean[j]
   *
   * This uses a row‑major traversal, which already matches the
   * underlying layout of `data`.
   * ----------------------------------------------------------*/
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];


  /* ------------------------------------------------------------
   * 3. Covariance of centered data.
   *
   * Mathematical definition:
   *   cov[i][j] = 1 / (float_n - 1) *
   *               sum_{k=0..N-1} data[k][i] * data[k][j]
   *
   * We exploit symmetry (cov[i][j] == cov[j][i]) and only
   * compute the upper triangle (j >= i), then copy.
   *
   * The computation is organized as a sequence of rank‑1
   * updates:
   *
   *   For each centered observation (row) x_k:
   *     cov += x_k^T * x_k  (upper triangle only)
   *
   * This streams each row of `data` exactly once and reuses
   * its elements for all (i, j) pairs in that row, in contrast
   * to the original (i, j, k) order which rescanned the
   * data columns for every (i, j) pair.
   * ----------------------------------------------------------*/

  /* Initialize upper triangle of covariance matrix to zero. */
  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      cov[i][j] = SCALAR_VAL(0.0);

  /* Accumulate unscaled covariance in upper triangle. */
  for (k = 0; k < _PB_N; k++)
    {
      /* Pointer to the current centered row: data[k][0..M-1].
         Using a local pointer gives the compiler a simpler
         base for strength reduction and vectorization. */
      DATA_TYPE *restrict data_k = data[k];

      int i0, j0;

      /* Block the feature dimensions (i, j). Blocking does not
         change the computed values; it only changes the order
         in which independent (i, j) pairs are visited. The
         triangular condition (j >= i) is preserved inside each
         block via the j_start guard. */
      for (i0 = 0; i0 < _PB_M; i0 += tile_m)
        {
          int i_max = (i0 + tile_m < _PB_M) ? i0 + tile_m : _PB_M;

          /* Restrict to tiles on or above the diagonal by
             starting j0 from i0. */
          for (j0 = i0; j0 < _PB_M; j0 += tile_m)
            {
              int j_max = (j0 + tile_m < _PB_M) ? j0 + tile_m : _PB_M;

              for (i = i0; i < i_max; i++)
                {
                  DATA_TYPE data_ki = data_k[i];

                  /* Enforce j >= i inside the (i0, j0) tile. */
                  int j_start = (i > j0) ? i : j0;

                  for (j = j_start; j < j_max; j++)
                    {
                      /* cov[i][j] += data[k][i] * data[k][j]; */
                      cov[i][j] += data_ki * data_k[j];
                    }
                }
            }
        }
    }

  /* Scale by 1 / (float_n - 1) and symmetrize. */
  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      {
        cov[i][j] *= inv_float_n_minus_one;
        cov[j][i] = cov[i][j];
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