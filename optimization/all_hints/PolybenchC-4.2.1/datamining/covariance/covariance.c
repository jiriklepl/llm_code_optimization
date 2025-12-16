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

/* Optional: OpenMP for parallel execution of the main kernel.
 * If the code is compiled without -fopenmp, the pragmas are ignored
 * by the compiler (as unknown pragmas) and the code remains valid.
 */
#ifdef _OPENMP
# include <omp.h>
#endif

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"

/* Tunable thresholds for enabling OpenMP parallelization.
 * These can be adapted for specific hardware / problem sizes.
 */
#ifndef COV_OMP_MIN_MEAN_WORK
# define COV_OMP_MIN_MEAN_WORK   1024L   /* ~ N*M threshold for mean computation */
#endif

#ifndef COV_OMP_MIN_CENTER_WORK
# define COV_OMP_MIN_CENTER_WORK 4096L   /* ~ N*M threshold for centering step  */
#endif

#ifndef COV_OMP_MIN_COV_WORK
# define COV_OMP_MIN_COV_WORK    64      /* ~ M threshold for covariance step   */
#endif


/* Array initialization. */
static
void init_array (int m, int n,
                 DATA_TYPE *float_n,
                 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  /* Keep original initialization order (i outer, j inner) to
   * preserve exact semantics; this loop is not performance-critical.
   */
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
   including the call and return. */
static
void kernel_covariance [[gnu::flatten, gnu::noinline]] (int m, int n,
                       DATA_TYPE float_n,
                       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
                       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
                       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  /* Local aliases with restrict-qualified pointers help the compiler
   * assume non-aliasing and generate better vectorized code.
   *
   *  data_ : N x M matrix, row-major
   *  cov_  : M x M matrix, row-major
   *  mean_ : length-M vector
   */
  DATA_TYPE (* restrict data_)[m] = data;
  DATA_TYPE (* restrict cov_)[m]  = cov;
  DATA_TYPE * restrict mean_      = mean;

  /* Effective loop bounds used by PolyBench. These can be smaller
   * than m and n depending on the compilation configuration.
   */
  const int M_iter = _PB_M;
  const int N_iter = _PB_N;

  /* Row stride (number of elements per row) in the data_ matrix. */
  const int stride = m;

  /* Precompute reciprocals to replace repeated divisions inside loops
   * with multiplications, which are significantly cheaper.
   */
  const DATA_TYPE inv_float_n =
      SCALAR_VAL(1.0) / float_n;
  const DATA_TYPE cov_scale =
      SCALAR_VAL(1.0) / (float_n - SCALAR_VAL(1.0));

#pragma scop
  /* --------------------------------------------------------------------
   * Step 1: Compute column means.
   *
   * Original code:
   *   for j
   *     mean[j] = 0;
   *     for i
   *       mean[j] += data[i][j];
   *     mean[j] /= float_n;
   *
   * We keep the same traversal order (j outer, i inner) to preserve
   * the accumulation order for each column, but:
   *   - fuse initialization, accumulation, and normalization for each j;
   *   - use a precomputed reciprocal (inv_float_n) instead of division;
   *   - parallelize the outer j-loop: each column is independent.
   * ------------------------------------------------------------------ */
#ifdef _OPENMP
  #pragma omp parallel for if ((long)M_iter * N_iter >= COV_OMP_MIN_MEAN_WORK) \
                          schedule(static) private(i)
#endif
  for (j = 0; j < M_iter; j++)
  {
    DATA_TYPE sum = SCALAR_VAL(0.0);
    for (i = 0; i < N_iter; i++)
      sum += data_[i][j];

    mean_[j] = sum * inv_float_n;
  }

  /* --------------------------------------------------------------------
   * Step 2: Center the data matrix by subtracting the column means.
   *
   * Original code:
   *   for i
   *     for j
   *       data[i][j] -= mean[j];
   *
   * We keep the same logical operation but:
   *   - traverse the matrix in row-major order (i outer, j inner);
   *   - use a row pointer to avoid repeated address computations;
   *   - parallelize over rows: each row is independent.
   * ------------------------------------------------------------------ */
#ifdef _OPENMP
  #pragma omp parallel for if ((long)N_iter * M_iter >= COV_OMP_MIN_CENTER_WORK) \
                          schedule(static) private(j)
#endif
  for (i = 0; i < N_iter; i++)
  {
    DATA_TYPE * restrict row = data_[i];
    for (j = 0; j < M_iter; j++)
      row[j] -= mean_[j];
  }

  /* --------------------------------------------------------------------
   * Step 3: Compute the covariance matrix.
   *
   * Original code:
   *   for i = 0..M-1
   *     for j = i..M-1
   *       cov[i][j] = 0;
   *       for k = 0..N-1
   *         cov[i][j] += data[k][i] * data[k][j];
   *       cov[i][j] /= (float_n - 1.0);
   *       cov[j][i] = cov[i][j];
   *
   * We preserve the same iteration space and accumulation order over k
   * for each (i,j) pair, but:
   *   - use local scalar accumulator 'sum' instead of repeatedly
   *     reading/writing cov_ inside the k-loop;
   *   - use pointer increments along the column (stride = m) to avoid
   *     repeated index arithmetic for data[k][i] and data[k][j];
   *   - use a precomputed scaling factor (cov_scale) instead of divide;
   *   - parallelize the outer i-loop: each (i,j) pair is independent.
   * ------------------------------------------------------------------ */
#ifdef _OPENMP
  #pragma omp parallel for if (M_iter >= COV_OMP_MIN_COV_WORK) \
                          schedule(static) private(j,k)
#endif
  for (i = 0; i < M_iter; i++)
  {
    for (j = i; j < M_iter; j++)
    {
      DATA_TYPE sum = SCALAR_VAL(0.0);

      /* Pointers walking down columns i and j.
       *   col_i points to data[0][i], then data[1][i], etc.
       *   col_j points to data[0][j], then data[1][j], etc.
       */
      DATA_TYPE * restrict col_i = &data_[0][i];
      DATA_TYPE * restrict col_j = &data_[0][j];

      /* No loop-carried dependencies: safe to vectorize.
       * The GCC 'ivdep' pragma tells the compiler there are no
       * inter-iteration dependencies preventing vectorization.
       */
      #pragma GCC ivdep
      for (k = 0; k < N_iter; k++)
      {
        sum += (*col_i) * (*col_j);
        col_i += stride;
        col_j += stride;
      }

      DATA_TYPE val = sum * cov_scale;

      cov_[i][j] = val;
      cov_[j][i] = val;
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