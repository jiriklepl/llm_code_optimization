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
from exo.API_scheduling import *
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# max_score(a, b) = max(a, b)
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
        return f"(({args[0]} >= {args[1]}) ? {args[0]} : {args[1]})"

    def globl(self, prim_type):
        return ""

    def interpret(self, args):
        return args[0] if args[0] >= args[1] else args[1]


max_score = _MaxScore()


# match(b1, b2) = 1 if (b1 + b2 == 3) else 0, cast to DATA_TYPE
class _Match(Extern):
    def __init__(self):
        super().__init__("match")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")
        t1 = args[0].type
        t2 = args[1].type
        if t1 != t2:
            raise _EErr("match: argument types must match")
        return t1

    def compile(self, args, prim_type):
        return (
            f"(({prim_type})((( {args[0]} + {args[1]} ) == 3) ? 1 : 0))"
        )

    def globl(self, prim_type):
        return ""

    def interpret(self, args):
        return type(args[0])(1) if (args[0] + args[1]) == 3 else type(args[0])(0)


match = _Match()


# Baseline dynamic-programming kernel, written in diagonal form for
# better locality: d = j - i is the subsequence length.
@proc
def kernel_nussinov_base(
    n: size,
    seq_buf: DATA_TYPE[n] @ DRAM,
    table: DATA_TYPE[n, n] @ DRAM,
):
    assert n >= 0

    # d is the subsequence length (j - i), from 1 up to n-1
    for d in seq(1, n):
        # i ranges so that j = i + d stays within [0, n-1]
        for i in seq(0, n - d):
            # j = i + d
            # Use a scalar accumulator for table[i, j] to reduce
            # repeated DRAM traffic on the DP table cell.
            q: DATA_TYPE

            # Start from the current value in the table (initialized to 0).
            q = table[i, i + d]

            # Case 1: base j is unpaired -> use [i, j-1].
            q = max_score(q, table[i, i + d - 1])

            # Case 2: base i is unpaired -> use [i+1, j].
            q = max_score(q, table[i + 1, i + d])

            # Case 3: i and j can pair (but disallow adjacent pairing).
            # For d == 1 (adjacent bases), we fall back to using only
            # table[i+1, j-1], exactly as in the original C code.
            if d > 1:
                q = max_score(
                    q,
                    table[i + 1, i + d - 1]
                    + match(seq_buf[i], seq_buf[i + d]),
                )
            else:
                q = max_score(q, table[i + 1, i + d - 1])

            # Case 4: bifurcation into [i, k] and [k+1, j].
            for k in seq(i + 1, i + d):
                q = max_score(
                    q,
                    table[i, k] + table[k + 1, i + d],
                )

            table[i, i + d] = q


# Scheduling: simple but effective scalar promotion.
# We (1) simplify expressions and control, and
# (2) lift the scalar accumulator `q` out of the inner i-loop so that
#     it is allocated once per d-iteration, not per i-iteration.
kernel_nussinov_opt = simplify(kernel_nussinov_base)

q_alloc = kernel_nussinov_opt.find("q : _")
kernel_nussinov_opt = lift_alloc(kernel_nussinov_opt, q_alloc)

# Export the scheduled kernel under the name expected by the C driver.
kernel_nussinov = kernel_nussinov_opt
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