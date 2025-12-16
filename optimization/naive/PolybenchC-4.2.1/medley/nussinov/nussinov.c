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

  /* base is AGCT / 0..3 */
  for (i = 0; i < n; i++) {
    seq[i] = (base)((i + 1) % 4);
  }

  /* Initialize DP table to zero.
     Use a simple nested loop for clarity and portability. */
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
   including the call and return.
   --------------------------------------------------------------------
   Optimizations applied (while preserving original semantics):
   - Use local restrict-qualified pointers to seq and table to give the
     compiler precise aliasing information.
   - Keep table[i][j] in a scalar 'best' (register) instead of repeatedly
     reading and writing table[i][j] from/to memory.
   - Replace 2D index arithmetic in the innermost k-loop with pointer
     arithmetic:
       * unit-stride access along row i for table[i][k]
       * stride-n access down column j for table[k+1][j]
     This removes repeated multiplications by 'n' in the innermost loop,
     improving computational efficiency and cache behavior.
   - Simplify boundary checks by exploiting the loop structure
     (e.g., j >= i+1 always holds in the inner loop).
   --------------------------------------------------------------------
*/
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov [[gnu::flatten, gnu::noinline]]
  (int n,
   base      POLYBENCH_1D(seq,  N, n),
   DATA_TYPE POLYBENCH_2D(table,N, N, n, n))
{
  /* Local restrict-qualified aliases: seq and table do not alias. */
  base      * restrict s = seq;
  DATA_TYPE (* restrict t)[n] = table;

  int i, j;

#pragma scop
  for (i = n - 1; i >= 0; i--) {
    /* Pointer to row i of the DP table, used heavily in the inner loops. */
    DATA_TYPE * restrict ti = t[i];
    base si = s[i];

    for (j = i + 1; j < n; j++) {

      /* Keep the current cell in a register. In the original code this
         starts from 0 because the table is zero-initialized and each
         (i,j) is visited exactly once. */
      DATA_TYPE best = ti[j];

      /* Case 1: j unpaired, inherit from (i, j-1).  j >= i+1 ⇒ j-1 >= 0. */
      {
        DATA_TYPE v = ti[j - 1]; /* table[i][j-1] */
        if (v > best)
          best = v;
      }

      /* Case 2: i unpaired, inherit from (i+1, j).
         Because j > i and i ranges down from n-1, here we always
         have i+1 < n whenever this loop body executes. */
      {
        DATA_TYPE v = t[i + 1][j]; /* table[i+1][j] */
        if (v > best)
          best = v;
      }

      /* Case 3: possible pair (i, j) plus the enclosed subsequence.
         Do not allow adjacent elements to bond; that is, only when
         j - i >= 2 do we add the match score. Otherwise we just
         propagate table[i+1][j-1], exactly as in the original code. */
      {
        DATA_TYPE v_inner = t[i + 1][j - 1]; /* table[i+1][j-1] */

        if (j - i >= 2) {
          /* i < j - 1 in the original condition. */
          DATA_TYPE score = v_inner + (DATA_TYPE)match(si, s[j]);
          if (score > best)
            best = score;
        } else {
          /* j == i+1 : adjacent, no bond allowed. */
          if (v_inner > best)
            best = v_inner;
        }
      }

      /* Case 4: bifurcation – split the subsequence at position k,
         with i < k < j. This is the O(n) inner loop and the main
         performance hotspot; we carefully structure memory accesses. */
      {
        int span = j - i - 1;  /* number of valid k in (i, j) */

        if (span > 0) {
          int k_start = i + 1;        /* first k */

          /* row_i_k walks along row i starting at column k_start
             (i.e., table[i][k_start]). */
          DATA_TYPE * restrict row_i_k = &ti[k_start];

          /* t_k1_j walks down column j starting at row k_start + 1
             (i.e., table[k_start+1][j]). We advance it by 'n' each
             step to move one row down while staying in column j. */
          DATA_TYPE * restrict t_k1_j  = &t[k_start + 1][j];

          for (int step = 0; step < span; ++step) {
            /* cand = table[i][k] + table[k+1][j] */
            DATA_TYPE cand = *row_i_k + *t_k1_j;
            if (cand > best)
              best = cand;

            ++row_i_k;   /* move to table[i][k+1]  */
            t_k1_j += n; /* move to table[(k+1)+1][j] */
          }
        }
      }

      /* Write the finalized value back to the DP table. */
      ti[j] = best;
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