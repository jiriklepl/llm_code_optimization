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
from exo.API_scheduling import *


# Tile size for the i-dimension (start index of the subsequence).
# This improves cache locality and creates coarse-grain work units.
I_TILE = 32


class _MaxScore(Extern):
    """
    max_score(a, b) = a if a >= b else b

    This is used instead of a data-dependent branch in Exo conditionals.
    """
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
    """
    match(b1, b2) = 1 if b1 and b2 form a canonical base pair, else 0.

    In the original PolyBench Nussinov kernel this is encoded as:
        ((b1 + b2) == 3 ? 1 : 0)
    where bases are integers in [0,3], and 1 is cast to DATA_TYPE.
    """
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
    """
    Optimized Nussinov kernel.

    - Uses standard wavefront (anti-diagonal) DP ordering:
          d = j - i  (subsequence length), 1 <= d < n
          i = 0 .. n - d - 1,  j = i + d
      which respects all data dependencies.

    - Tiles the i-dimension with a fixed tile size I_TILE to improve
      row-major cache locality for rows of 'table' and create coarse
      work units that can be parallelized in a schedule if desired.

    - Keeps the current DP cell table[i, j] in a scalar 'best', reducing
      memory traffic: one load at the start (from table[i, j-1]) and
      one store at the end.

    The recurrence matches the original C/Exo implementation:

        table[i, j] = max over:
            1) table[i, j-1]                     (j unpaired)
            2) table[i+1, j]                     (i unpaired)
            3) table[i+1, j-1] + match(seq[i], seq[j])
                   if j >= i+2 (no adjacent base pairs)
               or table[i+1, j-1]                if j == i+1
            4) max_{k in (i, j)} table[i, k] + table[k+1, j]   (bifurcation)
    """
    assert n >= 1

    # Outer loop over subsequence length d = j - i (the anti-diagonal index).
    for d in seq(1, n):
        # For a given d, valid i are 0 .. n-d-1 (inclusive), j = i + d.
        # We tile this i-range by I_TILE to improve locality in rows of 'table'.
        for I in seq(0, (n - d + I_TILE - 1) / I_TILE):
            for ii in seq(0, I_TILE):
                i = I * I_TILE + ii
                if i < n - d:
                    j = i + d

                    # Local accumulator holding the best score for interval [i, j].
                    # We start from the "j unpaired" case: table[i, j-1].
                    best: DATA_TYPE
                    best = table[i, j - 1]

                    # Case 2: i unpaired -> table[i+1, j].
                    best = max_score(best, table[i + 1, j])

                    # Case 3: possible pair (i, j), with adjacency restriction.
                    # Original code:
                    #   if i < j-1:
                    #       best = max(best, table[i+1, j-1] + match(seq[i], seq[j]));
                    #   else:
                    #       best = max(best, table[i+1, j-1]);
                    #
                    # Note that d = j - i; i < j-1  <=>  d > 1.
                    if d > 1:
                        best = max_score(
                            best,
                            table[i + 1, j - 1]
                            + match(seq_buf[i], seq_buf[j]),
                        )
                    else:
                        # Adjacent bases (d == 1) cannot pair; just propagate interior.
                        best = max_score(
                            best,
                            table[i + 1, j - 1],
                        )

                    # Case 4: bifurcation at position k, i < k < j.
                    # Original code:
                    #   for (k = i+1; k < j; k++)
                    #       best = max(best, table[i][k] + table[k+1][j]);
                    for k in seq(i + 1, j):
                        best = max_score(
                            best,
                            table[i, k] + table[k + 1, j],
                        )

                    # Commit the best score for interval [i, j].
                    table[i, j] = best


# Apply a semantics-preserving simplification pass to clean up the IR.
# This uses Exo's scheduling API (primitives) without changing the kernel's
# observable behavior, and may help the backend generate better C.
kernel_nussinov = simplify(kernel_nussinov)
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