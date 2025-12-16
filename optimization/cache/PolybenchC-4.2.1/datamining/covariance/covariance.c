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
   - Use `restrict` aliases for the main data structures to aid the optimizer.
   - Compute column means with row-major traversal for better cache usage.
   - Precompute reciprocal factors to replace divisions with multiplications.
   - Block the covariance computation over the column index `j` so that
     a small block of cov[i][j] is accumulated in registers before being
     written back, reducing memory traffic and improving data locality.
*/
static
void kernel_covariance [[gnu::flatten, gnu::noinline]] (int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  /* Local `restrict` aliases to help the compiler assume no aliasing
     between the three main arrays. */
  DATA_TYPE (* __restrict data_)[m] = data;
  DATA_TYPE (* __restrict cov_)[m]  = cov;
  DATA_TYPE * __restrict mean_      = mean;

  const int Mdim = _PB_M;
  const int Ndim = _PB_N;

  /* Precompute reciprocals once to avoid repeated divisions. */
  const DATA_TYPE inv_float_n = SCALAR_VAL(1.0) / float_n;
  const DATA_TYPE cov_scale =
    SCALAR_VAL(1.0) / (float_n - SCALAR_VAL(1.0));

  /* Tunable blocking factor for the covariance kernel.
     Compile-time override example: -DCOVARIANCE_J_BLOCK=64 */
#ifndef COVARIANCE_J_BLOCK
#define COVARIANCE_J_BLOCK 32
#endif

#pragma scop
  /* -------------------------------------------------------------
   * 1. Compute column means.
   *
   * Original code:
   *   for (j) { mean[j] = 0; for (i) mean[j] += data[i][j]; mean[j] /= n; }
   * traverses data column-wise, which is strided in row-major storage.
   *
   * Here we:
   *   - Initialize all means to zero.
   *   - Accumulate them row by row (i outer, j inner),
   *     so `data_[i][j]` is accessed contiguously in memory.
   *   - Scale by 1/n afterwards.
   * ------------------------------------------------------------- */

  /* Initialize means to zero. */
  for (j = 0; j < Mdim; ++j)
    mean_[j] = SCALAR_VAL(0.0);

  /* Accumulate column sums with row-major traversal. */
  for (i = 0; i < Ndim; ++i)
    {
      const DATA_TYPE * __restrict data_row = data_[i];
      for (j = 0; j < Mdim; ++j)
        mean_[j] += data_row[j];
    }

  /* Divide by n to obtain the mean. */
  for (j = 0; j < Mdim; ++j)
    mean_[j] *= inv_float_n;

  /* -------------------------------------------------------------
   * 2. Center the data: subtract the mean from each column.
   *    This loop is already row-major and cache-friendly; we keep the
   *    traversal order but use a row pointer for clarity and speed.
   * ------------------------------------------------------------- */
  for (i = 0; i < Ndim; ++i)
    {
      DATA_TYPE * __restrict data_row = data_[i];
      for (j = 0; j < Mdim; ++j)
        data_row[j] -= mean_[j];
    }

  /* -------------------------------------------------------------
   * 3. Compute the covariance matrix.
   *
   * Mathematical definition (for centered data):
   *   cov[i][j] = (1 / (n - 1)) * sum_{k=0..N-1} data[k][i] * data[k][j]
   * for all 0 <= i <= j < M, and cov[j][i] = cov[i][j].
   *
   * Baseline triple loop computes one (i,j) pair at a time and
   * re-reads column i for each partner column j.  Here we:
   *   - Keep the semantics identical,
   *   - For each fixed (i) and block of columns j in [jb, je),
   *     accumulate partial sums into a small local buffer `sum[]`,
   *   - For each (i, k) pair we load data[k][i] once per block,
   *     and reuse it across all j in that block.
   *
   * This reduces the number of loads of data[k][i] and lets the
   * compiler keep `sum` in registers/L1 cache until the block is done,
   * after which we scale and write results once to cov[i][j]
   * and its symmetric counterpart cov[j][i].
   * ------------------------------------------------------------- */

  for (i = 0; i < Mdim; ++i)
    {
      /* Process columns j in blocks [jb, j_max). */
      for (int jb = i; jb < Mdim; jb += COVARIANCE_J_BLOCK)
        {
          const int j_limit = jb + COVARIANCE_J_BLOCK;
          const int j_max   = (j_limit < Mdim) ? j_limit : Mdim;
          const int block_size = j_max - jb;

          /* Local partial sums for cov[i][j] in this j-block.
             Only the first `block_size` entries are used. */
#if defined(__GNUC__)
          DATA_TYPE sum[COVARIANCE_J_BLOCK] __attribute__((aligned(64)));
#else
          DATA_TYPE sum[COVARIANCE_J_BLOCK];
#endif

          /* Initialize the active part of the buffer. */
          for (j = 0; j < block_size; ++j)
            sum[j] = SCALAR_VAL(0.0);

          /* Accumulate over all rows k. */
          for (k = 0; k < Ndim; ++k)
            {
              const DATA_TYPE * __restrict data_row = data_[k];
              const DATA_TYPE dki = data_row[i]; /* column i at row k */

              /* Update sums for j in [jb, j_max). */
              for (j = 0; j < block_size; ++j)
                {
                  const int col_j = jb + j;
                  sum[j] += dki * data_row[col_j];
                }
            }

          /* Scale by 1/(n-1) and write back, enforcing symmetry. */
          for (j = 0; j < block_size; ++j)
            {
              const int col_j = jb + j;
              const DATA_TYPE v = sum[j] * cov_scale;
              cov_[i][col_j]   = v;
              cov_[col_j][i]   = v;
            }
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