/**
 * Exo Nussinov driver: mirrors PolyBench/C nussinov.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* Include the Exo-generated kernel header. */
#include "generated/nussinov/nussinov.h"

/* RNA bases represented as values in [0,3]. */
typedef DATA_TYPE base;

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
                 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  /* base is 0..3 */
  for (i = 0; i < n; i++) {
    seq[i] = (base)((i + 1) % 4);
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


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


class _MaxScore(Extern):
    def __init__(self):
        super().__init__("max_score")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")
        t1 = args[0].type
        t2 = args[1].type
        if t1 != t2:
            raise _EErr("max_score: argument types must match")
        return t1

    def compile(self, args, prim_type):
        # Returns the maximum of the two arguments.
        return f"(({args[0]} >= {args[1]}) ? {args[0]} : {args[1]})"

    def globl(self, prim_type):
        return ""

    def interpret(self, args):
        return args[0] if args[0] >= args[1] else args[1]


max_score = _MaxScore()


class _Match(Extern):
    def __init__(self):
        super().__init__("match")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")
        # Both arguments must have the same (numeric) type.
        t1 = args[0].type
        t2 = args[1].type
        if t1 != t2:
            raise _EErr("match: argument types must match")
        return t1

    def compile(self, args, prim_type):
        # Implements: ((b1 + b2) == 3 ? 1 : 0), cast to prim_type.
        return (
            f"(({prim_type})((( {args[0]} + {args[1]} ) == 3) ? 1 : 0))"
        )

    def globl(self, prim_type):
        return ""

    def interpret(self, args):
        return type(args[0])(1) if (args[0] + args[1]) == 3 else type(args[0])(0)


match = _Match()


@proc
def kernel_nussinov(
    n: size,
    seq_buf: DATA_TYPE[n] @ DRAM,
    table: DATA_TYPE[n, n] @ DRAM,
):
    # Original C loops:
    # for (i = n-1; i >= 0; i--)
    #   for (j = i+1; j < n; j++)
    #
    # We reindex with ii = 0..n-1, i = n-1-ii to get forward loops.
    for ii in seq(0, n):
        for j in seq(n - ii, n):
            # i = n - 1 - ii
            if j - 1 >= 0:
                table[n - 1 - ii, j] = max_score(
                    table[n - 1 - ii, j],
                    table[n - 1 - ii, j - 1],
                )

            if n - 1 - ii + 1 < n:
                table[n - 1 - ii, j] = max_score(
                    table[n - 1 - ii, j],
                    table[n - ii, j],
                )

            if j - 1 >= 0 and n - 1 - ii + 1 < n:
                # don't allow adjacent elements to bond
                if n - 1 - ii < j - 1:
                    table[n - 1 - ii, j] = max_score(
                        table[n - 1 - ii, j],
                        table[n - ii, j - 1]
                        + match(seq_buf[n - 1 - ii], seq_buf[j]),
                    )
                else:
                    table[n - 1 - ii, j] = max_score(
                        table[n - 1 - ii, j],
                        table[n - ii, j - 1],
                    )

            for k in seq(n - ii, j):
                table[n - 1 - ii, j] = max_score(
                    table[n - 1 - ii, j],
                    table[n - 1 - ii, k] + table[k + 1, j],
                )
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten views to 1D pointers. */
  kernel_nussinov(/*ctxt=*/NULL,
                  n,
                  (DATA_TYPE*)POLYBENCH_ARRAY(seq),
                  (DATA_TYPE*)POLYBENCH_ARRAY(table));

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