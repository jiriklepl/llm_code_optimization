/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* RNA bases represented as chars, range is [0,3] */
typedef char base;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Tunable tiling parameters for the kernel.
 * They can be overridden at compile time, e.g.:
 *   -DNUSSINOV_TILE_I=64 -DNUSSINOV_TILE_K=64
 */
#ifndef NUSSINOV_TILE_I
# define NUSSINOV_TILE_I 32
#endif

#ifndef NUSSINOV_TILE_K
# define NUSSINOV_TILE_K 32
#endif

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  // base is AGCT/0..3
  for (i = 0; i < n; i++) {
    seq[i] = (base) ((i + 1) % 4);
  }

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      table[i][j] = 0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))

{
  int i, j;
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
      t++;
    }
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov[[gnu::flatten, gnu::noinline]](int n, base POLYBENCH_1D(seq,N,n),
			   DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

  /* Optimized dynamic-programming kernel.
   *
   * Key changes relative to the original:
   *   - Iterate in "band" order using the distance d = j - i, which makes
   *     the dependency structure explicit and avoids redundant bounds checks.
   *   - Use a scalar accumulator `best` for table[i][j] to avoid repeated
   *     loads/stores of the same cell.
   *   - Tile the i dimension (NUSSINOV_TILE_I) to improve cache locality
   *     along rows.
   *   - Tile the k dimension (NUSSINOV_TILE_K) inside the split loop to
   *     improve locality for accesses to table[i][k] and table[k+1][j].
   *
   * Semantics are preserved exactly:
   *   table[i][j] = max of:
   *     - table[i][j-1]                     (j unpaired)
   *     - table[i+1][j]                     (i unpaired)
   *     - table[i+1][j-1] (+ match if j-i>1)
   *     - max_{k in (i, j)} table[i][k] + table[k+1][j]
   */
#pragma scop
  /* Distance from diagonal (subsequence length difference): d = j - i.
   * We only need d >= 1 because i < j.
   */
  for (int d = 1; d < _PB_N; d++) {
    const int max_i = _PB_N - d; /* i runs 0 .. max_i-1, j = i + d */

    /* Tile the i dimension to keep active rows in cache. */
    for (int ii = 0; ii < max_i; ii += NUSSINOV_TILE_I) {
      const int i_end = (ii + NUSSINOV_TILE_I < max_i) ? (ii + NUSSINOV_TILE_I) : max_i;

      for (i = ii; i < i_end; i++) {
        j = i + d;

        /* Row pointers for faster and more cache-friendly access.
         * Marked restrict to tell the compiler that these two rows do
         * not alias each other or any other row.
         */
        DATA_TYPE *restrict table_i   = table[i];
        DATA_TYPE *restrict table_ip1 = table[i + 1];

        /* Start from the current value (initialized to 0 in init_array). */
        DATA_TYPE best = table_i[j];

        /* Case 1: j is unpaired => use [i, j-1]. */
        DATA_TYPE tmp = table_i[j - 1];
        if (tmp > best)
          best = tmp;

        /* Case 2: i is unpaired => use [i+1, j]. */
        tmp = table_ip1[j];
        if (tmp > best)
          best = tmp;

        /* Case 3: potential pair (i, j).
         *
         * Original code:
         *   if (i < j-1)
         *     table[i][j] = max(table[i][j], table[i+1][j-1] + match(seq[i], seq[j]));
         *   else
         *     table[i][j] = max(table[i][j], table[i+1][j-1]);
         *
         * Using d = j - i:
         *   - i < j-1  <=>  d > 1
         *   - when d == 1, table[i+1][j-1] is a below-diagonal base case (0),
         *     so the "else" branch has no effect but is kept for exactness.
         */
        tmp = table_ip1[j - 1];
        if (d > 1) {
          const DATA_TYPE pair_score = (DATA_TYPE)match(seq[i], seq[j]); /* 0 or 1 */
          const DATA_TYPE cand = tmp + pair_score;
          if (cand > best)
            best = cand;
        } else {
          if (tmp > best)
            best = tmp;
        }

        /* Case 4: bifurcation over all split points k, i < k < j.
         * We tile k to control working-set size.
         */
        const int k_start = i + 1;
        const int k_stop  = j;
        for (int kk = k_start; kk < k_stop; kk += NUSSINOV_TILE_K) {
          const int kend = (kk + NUSSINOV_TILE_K < k_stop) ? (kk + NUSSINOV_TILE_K) : k_stop;

          for (k = kk; k < kend; k++) {
            const DATA_TYPE cand = table_i[k] + table[k + 1][j];
            if (cand > best)
              best = cand;
          }
        }

        /* Single write-back to the DP table cell. */
        table_i[j] = best;
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
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_nussinov (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  return 0;
}