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
  int i;

  /* base is AGCT / 0..3 */
  for (i = 0; i < n; i++) {
    /* (i + 1) is always non‑negative here, so & 3 is equivalent to % 4. */
    seq[i] = (base)((i + 1) & 3);
  }

  /* The DP table is allocated as a contiguous n x n block.
     Using memset improves initialization performance and cache use. */
  memset(&table[0][0], 0, (size_t)n * (size_t)n * sizeof(DATA_TYPE));
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

/* Mark as noinline so that the timing in main precisely measures this kernel.
   All optimizations are applied inside the function body. */
static void __attribute__((noinline))
kernel_nussinov(int n,
                base POLYBENCH_1D(seq,N,n),
                DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  /* Use the PolyBench loop bound for the iteration space,
     but keep the full logical size 'n' for the array shape. */

  /* Quick exit for degenerate problems (no work to do). */
  if (_PB_N <= 1)
    return;

  /* Create local restricted pointers to help the compiler with alias analysis.
     seq_ and table_ are the only pointers used to access the underlying data
     inside this kernel. */
  base * restrict seq_ = seq;
  DATA_TYPE (* restrict table_)[n] = table;

  int i, j, k;

#pragma scop
  /* Start from row _PB_N-2 because for i == _PB_N-1 the inner loop is empty. */
  for (i = _PB_N - 2; i >= 0; i--) {

    /* Cache row pointers for faster row-major access. */
    DATA_TYPE * restrict table_i   = table_[i];
    DATA_TYPE * restrict table_ip1 = table_[i + 1];
    base si = seq_[i];

    for (j = i + 1; j < _PB_N; j++) {

      /* Load the current cell once and update it in a register. */
      DATA_TYPE tij = table_i[j];

      /* Case 1: j is unpaired, take the best score up to j-1 in the same row. */
      DATA_TYPE v = table_i[j - 1];
      if (v > tij)
        tij = v;

      /* Case 2: i is unpaired, take the best score from row i+1 at column j. */
      v = table_ip1[j];
      if (v > tij)
        tij = v;

      /* Case 3: i pairs with j (if they are not adjacent). */
      DATA_TYPE v_base = table_ip1[j - 1];
      if (i < j - 1) {
        /* Don't allow adjacent elements to bond: only when i < j-1. */
        /* match() returns 0 or 1 depending on base compatibility. */
        int m = (((int)si + (int)seq_[j]) == 3) ? 1 : 0;
        v = v_base + (DATA_TYPE)m;
      } else {
        v = v_base;
      }
      if (v > tij)
        tij = v;

      /* Case 4: bifurcation: split between k and k+1. */
      DATA_TYPE best = tij;
      for (k = i + 1; k < j; k++) {
        /* table_i[k] is contiguous in memory; table_[k+1][j] uses the k+1-th row. */
        DATA_TYPE cand = table_i[k] + table_[k + 1][j];
        if (cand > best)
          best = cand;
      }

      /* Store the final best score back to the DP table. */
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