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

/* Tunable blocking factors for the (i,j) dimensions of the correlation
 * matrix.  These can be overridden at compile time, e.g.:
 *
 *   gcc -DCORR_BLOCK_I=64 -DCORR_BLOCK_J=64 ...
 *
 * to better match a particular cache hierarchy. */
#ifndef CORR_BLOCK_I
# define CORR_BLOCK_I 32
#endif

#ifndef CORR_BLOCK_J
# define CORR_BLOCK_J 32
#endif


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
   including the call and return. */
static
void kernel_correlation[[gnu::flatten, gnu::noinline]](int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  /* Local copies of the PolyBench loop bounds.  Using explicit variables
   * makes the intent clearer and allows them to be reused in multiple
   * places (e.g., as VLA dimensions) without repeating the macros. */
  const int Mdim = _PB_M;
  const int Ndim = _PB_N;

  DATA_TYPE eps = SCALAR_VAL(0.1);

  /* Temporary denominators used during normalization:
   *   denom[j] = sqrt(float_n) * stddev[j]
   * (computed after stddev[] is finalized). */
  DATA_TYPE denom[Mdim];

#pragma scop
  /* ------------------------------------------------------------- *
   * 1. Compute column means: mean[j] = (1/float_n) * sum_i data[i][j]
   *
   * The original code iterated with j as the outer loop and i as the
   * inner loop, resulting in column-wise (non-contiguous) accesses to
   * the row-major "data" array:
   *
   *   for (j) { mean[j]=0; for (i) mean[j] += data[i][j]; ... }
   *
   * Here we reorganize the loops to traverse "data" in row-major order
   * (i outer, j inner), which improves spatial locality.  For each
   * column j, the sequence of updates
   *
   *   mean[j] += data[0][j];
   *   mean[j] += data[1][j];
   *   ...
   *   mean[j] += data[N-1][j];
   *
   * is preserved; we merely interleave updates to different columns.
   * ------------------------------------------------------------- */

  /* Initialize all means to 0.0. */
  for (j = 0; j < Mdim; j++)
    mean[j] = SCALAR_VAL(0.0);

  /* Accumulate column sums with row-major traversal. */
  for (i = 0; i < Ndim; i++)
  {
    DATA_TYPE *restrict row = data[i];
    for (j = 0; j < Mdim; j++)
      mean[j] += row[j];
  }

  /* Divide by float_n to obtain the mean.  We keep the division here
   * to preserve the numerical semantics of the original code. */
  for (j = 0; j < Mdim; j++)
    mean[j] /= float_n;


  /* ------------------------------------------------------------- *
   * 2. Compute standard deviations:
   *
   *    stddev[j] = sqrt( (1/float_n) * sum_i (data[i][j] - mean[j])^2 )
   *
   * As above, we use a row-major traversal (i outer, j inner) while
   * preserving, for each j, the exact sequence of accumulations over i.
   * ------------------------------------------------------------- */

  /* Initialize all standard deviations to 0.0. */
  for (j = 0; j < Mdim; j++)
    stddev[j] = SCALAR_VAL(0.0);

  /* Accumulate squared differences. */
  for (i = 0; i < Ndim; i++)
  {
    DATA_TYPE *restrict row = data[i];
    for (j = 0; j < Mdim; j++)
    {
      DATA_TYPE diff = row[j] - mean[j];
      stddev[j] += diff * diff;
    }
  }

  /* Finalize stddev: divide by float_n, take square root, and clamp
   * small values to 1.0 to avoid divisions by (near) zero later. */
  for (j = 0; j < Mdim; j++)
  {
    stddev[j] /= float_n;
    stddev[j] = SQRT_FUN(stddev[j]);
    /* The following is an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero-
       divide. */
    stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
  }


  /* ------------------------------------------------------------- *
   * 3. Center and normalize the column vectors:
   *
   *    data[i][j] <- (data[i][j] - mean[j]) / (sqrt(float_n)*stddev[j])
   *
   * We first precompute one denominator per column:
   *
   *    denom[j] = sqrt(float_n) * stddev[j]
   *
   * which avoids recomputing the product inside the inner loops.
   * Each element is still divided by denom[j], so the numerical
   * semantics of the original implementation are preserved.
   * ------------------------------------------------------------- */

  DATA_TYPE sqrt_float_n = SQRT_FUN(float_n);

  for (j = 0; j < Mdim; j++)
    denom[j] = sqrt_float_n * stddev[j];

  /* Apply centering and scaling with row-major traversal. */
  for (i = 0; i < Ndim; i++)
  {
    DATA_TYPE *restrict row = data[i];
    for (j = 0; j < Mdim; j++)
    {
      row[j] -= mean[j];
      row[j] /= denom[j];
    }
  }


  /* ------------------------------------------------------------- *
   * 4. Compute the m x m correlation matrix.
   *
   * Original algorithm:
   *
   *   for (i = 0; i < M-1; i++) {
   *     corr[i][i] = 1;
   *     for (j = i+1; j < M; j++) {
   *       corr[i][j] = 0;
   *       for (k = 0; k < N; k++)
   *         corr[i][j] += data[k][i] * data[k][j];
   *       corr[j][i] = corr[i][j];
   *     }
   *   }
   *   corr[M-1][M-1] = 1;
   *
   * This performs, for each pair (i,j), a full pass over the N rows,
   * repeatedly streaming the same rows of "data" from memory.
   *
   * Optimized algorithm:
   *   - View "corr" as the Gram matrix of the normalized data:
   *         corr = data^T * data
   *   - For each row k, perform a symmetric rank‑1 update:
   *         corr += data[k]^T * data[k]
   *   - Use explicit blocking in (i,j) to improve cache locality
   *     for both "data" (row-major) and "corr".
   *   - Only the upper triangular part (i <= j) is accumulated;
   *     the lower triangular part is filled by symmetry at the end.
   *
   * For every pair (i,j) with i <= j, the sequence of updates over k
   * is still performed in increasing k order, as in the original
   * implementation; we only change the way work is organized across
   * different (i,j) pairs to improve locality.
   * ------------------------------------------------------------- */

  const int i_block_size = CORR_BLOCK_I;
  const int j_block_size = CORR_BLOCK_J;

  /* Initialize the entire correlation matrix to 0.0.  The lower
   * triangle will be overwritten later by symmetry; initializing
   * it here avoids any risk of reading uninitialized values. */
  for (i = 0; i < Mdim; i++)
    for (j = 0; j < Mdim; j++)
      corr[i][j] = SCALAR_VAL(0.0);

  /* Blocked accumulation of the upper triangular part of corr. */
  for (int ii = 0; ii < Mdim; ii += i_block_size)
  {
    int i_max = ii + i_block_size;
    if (i_max > Mdim)
      i_max = Mdim;

    for (int jj = ii; jj < Mdim; jj += j_block_size)
    {
      int j_max = jj + j_block_size;
      if (j_max > Mdim)
        j_max = Mdim;

      for (k = 0; k < Ndim; k++)
      {
        DATA_TYPE *restrict row = data[k];

        for (i = ii; i < i_max; i++)
        {
          DATA_TYPE data_ki = row[i];

          /* Only update the upper triangle: j >= i.
           * For off-diagonal blocks (jj > ii), jj may be greater than i. */
          int j_start = (jj > i) ? jj : i;

          for (j = j_start; j < j_max; j++)
            corr[i][j] += data_ki * row[j];
        }
      }
    }
  }

  /* Enforce symmetry and set diagonal elements to 1.0, exactly as in the
   * original implementation.  Any values accumulated on the diagonal
   * (which, mathematically, should be very close to 1.0) are explicitly
   * overwritten to match the original behavior. */
  for (i = 0; i < Mdim; i++)
  {
    corr[i][i] = SCALAR_VAL(1.0);
    for (j = i+1; j < Mdim; j++)
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