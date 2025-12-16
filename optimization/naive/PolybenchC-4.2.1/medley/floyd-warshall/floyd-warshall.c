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


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
  {
    /* Cache the row pointer to improve locality and reduce
       address calculation inside the inner loop. */
    DATA_TYPE* row_i = path[i];

    for (j = 0; j < n; j++) {
      row_i[j] = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        row_i[j] = 999;
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
  {
    /* Again, cache row pointer for better locality. */
    DATA_TYPE* row_i = path[i];

    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, row_i[j]);
    }
  }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void __attribute__((noinline))
kernel_floyd_warshall(int n,
                      DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  /*
   * Optimized Floyd–Warshall:
   *
   * - Keep the original (k, i, j) loop nest and iteration order to
   *   preserve semantics exactly.
   * - For each fixed (k, i), the value path[i][k] is reused many
   *   times in the original code (once per j), causing redundant
   *   loads.  We split the j-loop around j == k so that we can safely
   *   hoist and reuse path[i][k] without changing behavior:
   *
   *     * For j < k: path[i][k] has not been updated yet; we use the
   *       original (old) value.
   *     * For j == k: we update path[i][k] using the old value.
   *     * For j > k: we must see the updated value of path[i][k].
   *
   *   This exactly matches the sequential semantics of the original
   *   triple loop (including corner cases such as negative weights),
   *   but reduces the number of loads of path[i][k] from O(N) per
   *   (i,k) pair to O(1) per (i,k) pair.
   *
   * - Row pointers (row_i, row_k) are cached to reduce address
   *   computation and to help the compiler generate tight, vectorized
   *   inner loops with good data locality.
   */
  for (k = 0; k < _PB_N; k++)
  {
    /* Cache the k-th row once per outer iteration.  Note: not 'restrict'
       because for i == k, row_k and row_i alias the same row. */
    DATA_TYPE* row_k = path[k];

    for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE* row_i = path[i];

      /* Value path[i][k] before any update at this (i,k). */
      DATA_TYPE path_ik = row_i[k];

      /* 1) Handle j < k with the original (old) path[i][k]. */
      int j_end1 = k;
      /* This loop has no loop-carried dependencies and is safe to
         vectorize. The compiler should be able to do this at -O3. */
      for (j = 0; j < j_end1; j++)
      {
        DATA_TYPE cur   = row_i[j];
        DATA_TYPE via_k = path_ik + row_k[j];
        row_i[j] = (cur < via_k) ? cur : via_k;
      }

      /* 2) Handle j == k: this is the only iteration that can update
         path[i][k]. We use the old path_ik on both sides, then store
         the new value back and keep it in path_ik for later reuse. */
      {
        DATA_TYPE cur   = path_ik;      /* old row_i[k] */
        DATA_TYPE via_k = path_ik + row_k[k];
        DATA_TYPE new_ik = (cur < via_k) ? cur : via_k;

        row_i[k] = new_ik;
        path_ik  = new_ik; /* updated path[i][k] for j > k */
      }

      /* 3) Handle j > k with the updated path[i][k]. Again, this loop
         is free of loop-carried dependencies and vectorizable. */
      for (j = k + 1; j < _PB_N; j++)
      {
        DATA_TYPE cur   = row_i[j];
        DATA_TYPE via_k = path_ik + row_k[j];
        row_i[j] = (cur < via_k) ? cur : via_k;
      }
    }
  }
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