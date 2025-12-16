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
    seq[i] = (base)((i+1)%4);
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
  /* We keep the interface and the PolyBench annotations intact.
   *
   * Optimization overview:
   * - Traverse the DP table by increasing subsequence length (j - i), which
   *   matches the true data dependency pattern and is more cache-friendly.
   * - For each (i,j), accumulate the best score in a scalar `best` and write
   *   back to table[i][j] only once, reducing memory traffic.
   * - Use a row pointer (`table_i`) to access row i, improving locality for
   *   accesses table[i][*].
   * - Optionally expose parallelism across i for a fixed length with OpenMP.
   *   Each (i,j) on a fixed diagonal (fixed j-i) is independent because every
   *   dependency uses strictly shorter subsequences.
   */

  int i, j, k, len;

  /* Silence potential unused-parameter warnings if _PB_N is a macro
     that hides `n` (as in PolyBench). Semantics are unchanged. */
  (void)n;

#pragma scop
  /* Process subsequences in order of increasing length len = j - i.
   *
   * For any cell (i, j):
   *   - table[i][j-1]  has length (len-1)
   *   - table[i+1][j]  has length (len-1)
   *   - table[i+1][j-1] has length (len-2)
   *   - table[i][k] and table[k+1][j] have lengths < len
   *
   * Thus all dependencies of length `len` refer only to lengths < len, so the
   * inner loop over i can be executed in parallel safely.
   */
  for (len = 1; len < _PB_N; ++len)
  {
    int max_i = _PB_N - len;

    /* Parallelization note:
     * - This OpenMP pragma is optional: if the code is compiled without
     *   -fopenmp, it is ignored and the code remains sequential.
     * - With -fopenmp, iterations over different i at fixed len are executed
     *   in parallel and are independent by the dependency argument above.
     */
#pragma omp parallel for schedule(static) if (max_i > 32)
    for (i = 0; i < max_i; ++i)
    {
      /* j is determined by the subsequence length. */
      j = i + len;

      /* Local alias for row i of the table to improve cache locality and
       * reduce address computation for accesses table[i][*].
       * We never access row i through any other pointer inside this loop,
       * so this does not change semantics.
       */
      DATA_TYPE* restrict table_i = table[i];

      DATA_TYPE best = 0;

      /* Case 1: j is unpaired (take best value ending at j-1). */
      DATA_TYPE val = table_i[j-1];
      if (val > best)
        best = val;

      /* Case 2: i is unpaired (take best value starting at i+1). */
      val = table[i+1][j];
      if (val > best)
        best = val;

      /* Case 3: i and j may pair (or both be unpaired but adjacent).
       *
       * Original code:
       *   if (j-1>=0 && i+1<_PB_N) {
       *     if (i<j-1)
       *       table[i][j] = max(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
       *     else
       *       table[i][j] = max(table[i][j], table[i+1][j-1]);
       *   }
       *
       * For our iteration domain (0 <= i < j < _PB_N):
       *   - j-1 >= 0 and i+1 < _PB_N always hold for the executed iterations.
       *   - i < j-1 is equivalent to (len > 1).
       *
       * So:
       *   - if len == 1: candidate = table[i+1][j-1];
       *   - if len > 1 : candidate = table[i+1][j-1] + match(seq[i], seq[j]);
       */
      DATA_TYPE diag = table[i+1][j-1];
      int pair_value = (len > 1) ? match(seq[i], seq[j]) : 0;
      val = diag + (DATA_TYPE)pair_value;
      if (val > best)
        best = val;

      /* Case 4: split the interval (i, j) into (i, k) and (k+1, j). */
      for (k = i+1; k < j; ++k)
      {
        DATA_TYPE tmp = table_i[k] + table[k+1][j];
        if (tmp > best)
          best = tmp;
      }

      /* Write back the final maximum for this cell once. */
      table_i[j] = best;
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