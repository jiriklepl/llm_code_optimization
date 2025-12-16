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

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  // base is AGCT/0..3
  for (i = 0; i < n; i++) {
    seq[i] = (base)((i+1) % 4);
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
void kernel_nussinov[[gnu::flatten, gnu::noinline]](int n,
                                   base POLYBENCH_1D(seq,N,n),
			           DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  /* We use the PolyBench-provided loop bound so that this remains
     compatible with PolyBench's instrumentation and validation. */
  const int N = _PB_N;

  int i, j, k;

#pragma scop
  /* ------------------------------------------------------------------
   * Optimized dynamic-programming traversal
   *
   * The original code iterated as:
   *
   *   for (i = N-1; i >= 0; --i)
   *     for (j = i+1; j < N; ++j)
   *
   * This processes the upper triangle row by row.
   *
   * The DP dependencies only refer to subproblems of strictly smaller
   * (j-i) “length”, so we can safely reorder the computation to iterate
   * by increasing subsequence length L = j-i:
   *
   *   for (L = 1; L < N; ++L)
   *     for (i = 0; i < N-L; ++i) { j = i+L; ... }
   *
   * For fixed L, all (i,j) with j-i == L are independent of each other
   * and depend only on entries with smaller length. This improves data
   * locality (we touch neighboring diagonals in order) and exposes a
   * natural parallelism opportunity on the inner i-loop.
   * ------------------------------------------------------------------ */
  for (int L = 1; L < N; ++L)
  {
    /* Parallelize across cells on the same diagonal (same L).
       There are no loop-carried dependencies along i for a fixed L,
       since all references are to strictly shorter subsequences.     */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < N - L; ++i)
    {
      j = i + L;

      /* Row pointers improve cache locality and reduce address
         calculation overhead versus repeated table[i][*] access.   */
      DATA_TYPE *restrict table_i   = table[i];
      DATA_TYPE *restrict table_ip1 = table[i+1];

      /* Accumulator for the best score for (i,j).
         We keep it in a register and write back once at the end to
         minimize memory traffic and allow better optimization.      */
      DATA_TYPE q = table_i[j]; /* initially zero from init_array */

      /* The bounds j-1 >= 0 and i+1 < N always hold here:
           - L >= 1  =>  j = i+L >= i+1  =>  j-1 >= i >= 0
           - j <= N-1 => i = j-L <= N-2  =>  i+1 <= N-1
         so we can safely drop the original bound checks.            */

      /* Option 1: skip j (rightmost base) */
      DATA_TYPE v = table_i[j-1];
      if (v > q) q = v;

      /* Option 2: skip i (leftmost base) */
      v = table_ip1[j];
      if (v > q) q = v;

      /* Option 3: pair i and j (if not adjacent) and take
         table[i+1][j-1] + match(seq[i], seq[j]).
         For L == 1 (adjacent), pairing is not allowed and we simply
         use table[i+1][j-1], which is in the lower triangle and
         remains 0 as in the original code.                           */
      const DATA_TYPE t_diag = table_ip1[j-1]; /* table[i+1][j-1] */
      DATA_TYPE pair_score = 0;

      if (L > 1) {
        const base si = seq[i];
        const base sj = seq[j];
        pair_score = match(si, sj);
      }

      v = t_diag + pair_score;
      if (v > q) q = v;

      /* Option 4: bifurcation:
         max over i < k < j of table[i][k] + table[k+1][j].

         We manually unroll the inner loop by a factor of 4 to
         increase instruction-level parallelism and provide a simple
         tunable knob (the unroll factor) for different hardware.
         The remainder iterations are handled in a cleanup loop.      */

      const int k_start = i + 1;
      const int k_end   = j;          /* exclusive upper bound */
      const int count   = k_end - k_start;
      const int k_unroll_end = k_start + (count & ~3); /* multiple of 4 */

      /* Unrolled loop: handle blocks of 4 ks at a time. */
      for (k = k_start; k < k_unroll_end; k += 4)
      {
        /* Each candX corresponds to a different split point. */
        DATA_TYPE cand0 = table_i[k]     + table[k+1][j];
        if (cand0 > q) q = cand0;

        DATA_TYPE cand1 = table_i[k+1]   + table[k+2][j];
        if (cand1 > q) q = cand1;

        DATA_TYPE cand2 = table_i[k+2]   + table[k+3][j];
        if (cand2 > q) q = cand2;

        DATA_TYPE cand3 = table_i[k+3]   + table[k+4][j];
        if (cand3 > q) q = cand3;
      }

      /* Remainder loop for any leftover k values (< 4). */
      for (; k < k_end; ++k)
      {
        DATA_TYPE cand = table_i[k] + table[k+1][j];
        if (cand > q) q = cand;
      }

      /* Store the final best score for subsequence (i,j). */
      table_i[j] = q;
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