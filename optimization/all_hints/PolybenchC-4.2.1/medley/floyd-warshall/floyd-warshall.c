/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"

#ifdef _OPENMP
# include <omp.h>
#endif


/* Array initialization.
 *
 * This is not the dominant cost, but we still improve locality by
 * caching the row pointer in the inner loop.
 */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++) {
    DATA_TYPE *row = path[i];
    for (j = 0; j < n; j++) {
      DATA_TYPE val = (DATA_TYPE)((i * j) % 7 + 1);
      int ipj = i + j;
      if (ipj % 13 == 0 || ipj % 7 == 0 || ipj % 11 == 0)
        val = (DATA_TYPE)999;
      row[j] = val;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* --------------------------------------------------------------------
 * Optimized Floyd–Warshall kernel
 *
 * Key ideas:
 *  - Keep the canonical loop order with k outermost to preserve the
 *    dynamic-programming semantics.
 *  - For each (k, i), process the entire row i in a helper function
 *    that:
 *      * Splits the j-loop around j == k to safely cache path[i][k]
 *        while exactly preserving the original semantics, even for
 *        arbitrary inputs (including negative weights and NaNs).
 *      * Uses a cached left-hand term (path[i][k]) to reduce loads.
 *  - Parallelize over i for each fixed k while respecting the true
 *    dependency through row k:
 *      * All rows i < k are computed first (they must see the "old"
 *        row k).
 *      * Then row k itself is updated.
 *      * Finally rows i > k are computed, seeing the updated row k.
 *    This reproduces the original sequential dependence pattern exactly.
 *  - Optionally use OpenMP 'simd' to encourage vectorization of j-loops.
 *
 * The helper function fw_process_row implements the exact scalar
 * semantics of the original inner j-loop, including the precise
 * behavior of the ?: operator with respect to NaNs.
 * ------------------------------------------------------------------*/

/* Process one row "i" for a fixed intermediate vertex "k".
 *
 * Original scalar update for each (i, j, k):
 *
 *   path[i][j] = path[i][j] < path[i][k] + path[k][j]
 *                ? path[i][j]
 *                : path[i][k] + path[k][j];
 *
 * Note: the correct transformation of this expression to an "if"
 * form that preserves IEEE-754 NaN behavior is:
 *
 *   tmp_old = path[i][j];
 *   tmp_new = path[i][k] + path[k][j];
 *   if (!(tmp_old < tmp_new))
 *     path[i][j] = tmp_new;
 *
 * because the original assigns tmp_new when (tmp_old < tmp_new) is
 * false, which is also the case when tmp_new is NaN.
 *
 * We also have a subtle dependency when j == k: path[i][k] appears
 * on both sides of the assignment, and for subsequent j > k we must
 * use the (possibly) updated path[i][k]. To cache path[i][k] while
 * preserving correctness for all inputs, we:
 *
 *   - Capture path[i][k] into 'dik_before'.
 *   - Use dik_before for all j < k and for computing the j == k
 *     update.
 *   - After j == k, read back path[i][k] as 'dik_after', and use
 *     this for all j > k.
 *
 * This exactly matches the behavior of the original single j-loop.
 */
static inline
void fw_process_row(int n,
                    DATA_TYPE *row_i,
                    DATA_TYPE *row_k,
                    int k)
{
  const DATA_TYPE dik_before = row_i[k];
  int j;

  /* j from 0 to k-1: path[i][k] is still the original value (dik_before),
   * and path[i][k] is not written in this range, so using the cached
   * value is exact.
   */
#ifdef _OPENMP
# pragma omp simd
#endif
  for (j = 0; j < k; ++j)
  {
    DATA_TYPE dij = row_i[j];
    DATA_TYPE alt = dik_before + row_k[j];
    /* Equivalent to:
     *   row_i[j] = (dij < alt) ? dij : alt;
     * with correct NaN handling.
     */
    if (!(dij < alt))
      row_i[j] = alt;
  }

  /* j == k: this is the only point where path[i][k] can change
   * during this k-iteration. We must use the original path[i][k]
   * value (dik_before) on the right-hand side, and compare using
   * the current left-hand value as in the original code.
   */
  {
    DATA_TYPE dij = row_i[k];
    DATA_TYPE alt = dik_before + row_k[k];
    if (!(dij < alt))
      row_i[k] = alt;
  }

  /* For j > k we must use the (possibly) updated path[i][k]. */
  const DATA_TYPE dik_after = row_i[k];

#ifdef _OPENMP
# pragma omp simd
#endif
  for (j = k + 1; j < n; ++j)
  {
    DATA_TYPE dij = row_i[j];
    DATA_TYPE alt = dik_after + row_k[j];
    if (!(dij < alt))
      row_i[j] = alt;
  }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall[[gnu::flatten, gnu::noinline]](int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, k;
  const int n_ = _PB_N;

#pragma scop

#ifdef _OPENMP
  /* Single parallel region to amortize thread creation.
   *
   * For each k we respect the dependencies via row k:
   *   - First, all rows i < k are updated in parallel; they see
   *     the "old" row_k, as in the sequential code because row k
   *     has not yet been processed.
   *   - Next, row k itself is updated by a single thread.
   *   - Finally, rows i > k are updated in parallel; they see the
   *     updated row_k, exactly matching the sequential ordering.
   */
# pragma omp parallel private(i, k)
  {
    for (k = 0; k < n_; ++k)
    {
      DATA_TYPE *row_k = path[k];

      /* Group 1: i in [0, k) — must see old row_k. */
#     pragma omp for schedule(static)
      for (i = 0; i < k; ++i)
      {
        DATA_TYPE *row_i = path[i];
        fw_process_row(n_, row_i, row_k, k);
      }

      /* Row k itself — performed once, after all i < k.
       * 'single' has an implicit barrier at the end, ensuring that
       * all threads see the updated row_k before proceeding.
       */
#     pragma omp single
      {
        DATA_TYPE *row_i = path[k];
        fw_process_row(n_, row_i, row_k, k);
      }

      /* Group 2: i in (k, n_) — must see updated row_k. */
#     pragma omp for schedule(static)
      for (i = k + 1; i < n_; ++i)
      {
        DATA_TYPE *row_i = path[i];
        fw_process_row(n_, row_i, row_k, k);
      }
    }
  }

#else /* !_OPENMP: purely sequential, but keeping all locality
        * and arithmetic improvements. */
  for (k = 0; k < n_; k++)
  {
    DATA_TYPE *row_k = path[k];
    for (i = 0; i < n_; i++)
    {
      DATA_TYPE *row_i = path[i];
      fw_process_row(n_, row_i, row_k, k);
    }
  }
#endif

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}