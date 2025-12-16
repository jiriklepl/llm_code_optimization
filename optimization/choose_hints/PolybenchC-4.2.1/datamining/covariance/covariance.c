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
   - Mean computation rewritten to traverse data row-wise (contiguous in
     memory), improving cache locality and enabling better vectorization.
   - Mean scaling and covariance scaling use precomputed reciprocals to
     replace many divisions with multiplications.
   - Local `restrict`-qualified pointers are used to help the compiler
     assume non-aliasing between the main arrays.
   - Covariance computation keeps the original mathematical formulation
     but:
       * precomputes columns pointers to express regular strides,
       * exploits symmetry (i,j) == (j,i) as in the original code,
       * is optionally parallelized across the outer-most loop with
         OpenMP (if compiled with -fopenmp; otherwise pragmas are ignored).
 */
static
void kernel_covariance[[gnu::flatten, gnu::noinline]](int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  /* Create local VLA pointers with explicit `restrict` to aid
     the optimizer.  The memory layout is still exactly the same;
     we only give the compiler more information. */
  DATA_TYPE (*restrict dataR)[m] = (DATA_TYPE (*)[m])data;
  DATA_TYPE (*restrict covR)[m]  = (DATA_TYPE (*)[m])cov;
  DATA_TYPE *restrict meanR      = (DATA_TYPE *)mean;

  const int pb_n = _PB_N;
  const int pb_m = _PB_M;

#pragma scop
  /* --------------------------------------------------------------------
   * Step 1: Compute mean of each column.
   *
   * Original code:
   *   for (j)
   *     mean[j] = 0;
   *     for (i) mean[j] += data[i][j];
   *     mean[j] /= float_n;
   *
   * That iterates column-wise (stride m) which is cache-unfriendly for
   * row-major storage.  We keep the same mathematical result but
   * accumulate using a row-major traversal: for each row, update all
   * column sums.  This scans each row contiguously once.
   * ------------------------------------------------------------------*/

  /* Initialize means to zero. */
  for (j = 0; j < pb_m; j++)
    meanR[j] = SCALAR_VAL(0.0);

  /* Accumulate column sums with row-major traversal. */
  for (i = 0; i < pb_n; i++)
    {
      DATA_TYPE *restrict row = dataR[i];
      /* Each iteration touches row[0..m-1] contiguously. */
#pragma GCC ivdep
      for (j = 0; j < pb_m; j++)
        meanR[j] += row[j];
    }

  /* Scale by 1/float_n.  Precompute reciprocal to avoid many divisions. */
  const DATA_TYPE inv_float_n = SCALAR_VAL(1.0) / float_n;
  for (j = 0; j < pb_m; j++)
    meanR[j] *= inv_float_n;


  /* --------------------------------------------------------------------
   * Step 2: Center the data by subtracting the mean from each column.
   *
   * Original code was already row-major:
   *   for (i) for (j) data[i][j] -= mean[j];
   * We keep that structure and only use local pointers and `restrict`
   * for clarity and potential optimization.
   * ------------------------------------------------------------------*/
  for (i = 0; i < pb_n; i++)
    {
      DATA_TYPE *restrict row = dataR[i];
#pragma GCC ivdep
      for (j = 0; j < pb_m; j++)
        row[j] -= meanR[j];
    }


  /* --------------------------------------------------------------------
   * Step 3: Compute covariance matrix.
   *
   * Original code:
   *   for (i)
   *     for (j=i..M-1)
   *       cov[i][j] = 0;
   *       for (k) cov[i][j] += data[k][i] * data[k][j];
   *       cov[i][j] /= (float_n - 1);
   *       cov[j][i] = cov[i][j];
   *
   * We preserve this computation but:
   *   - Precompute scale = 1/(float_n - 1) (fewer divisions).
   *   - Use pointer arithmetic to express the strided access pattern
   *     to columns explicitly (one pointer per column advanced by `m`).
   *   - Parallelize the outer loop over i with OpenMP; each (i,j) pair
   *     is computed exactly once and is independent.
   * ------------------------------------------------------------------*/

  const DATA_TYPE scale = SCALAR_VAL(1.0) / (float_n - SCALAR_VAL(1.0));

  /* Parallelization is optional: if compiled without -fopenmp,
     this pragma is ignored and execution remains sequential. */
#pragma omp parallel for schedule(static) private(j,k) if (pb_m > 1)
  for (i = 0; i < pb_m; i++)
    {
      for (j = i; j < pb_m; j++)
        {
          /* Pointers to the start of columns i and j.  We step by `m`
             (the row length) to move down each column. */
          const DATA_TYPE *col_i = &dataR[0][i];
          const DATA_TYPE *col_j = &dataR[0][j];

          DATA_TYPE sum = SCALAR_VAL(0.0);

#pragma GCC ivdep
          for (k = 0; k < pb_n; k++)
            {
              sum += (*col_i) * (*col_j);
              col_i += m;
              col_j += m;
            }

          DATA_TYPE val = sum * scale;

          /* Exploit symmetry as in the original code. */
          covR[i][j] = val;
          covR[j][i] = val;
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