/**
 * Exo Floyd-Warshall driver: mirrors PolyBench/C floyd-warshall.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"

/* Include the Exo-generated kernel header. */
#include "generated/floyd-warshall/floyd-warshall.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
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

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *
from exo.core.extern import Extern, _EErr


class _Min(Extern):
    def __init__(self):
        super().__init__("min")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")

        if args[0].type != args[1].type:
            raise _EErr(
                f"expected both arguments to have the same type, "
                f"but got {args[0].type} and {args[1].type}"
            )

        return args[0].type

    def compile(self, args, prim_type):
        # args[0], args[1] are C strings for the two arguments
        # prim_type is the underlying C primitive type (e.g., float, double, int32_t)
        return f"(({prim_type})(({args[0]} < {args[1]}) ? {args[0]} : {args[1]}))"

    def globl(self, prim_type):
        # No additional global code needed
        return ""

    def interpret(self, args):
        return args[0] if args[0] < args[1] else args[1]


min = _Min()


@proc
def kernel_floyd_warshall(
    n: size,
    path: DATA_TYPE[n, n] @ DRAM,
):
    # Temporary buffer to stage one row of `path` (row k).
    # This is O(n) extra storage and greatly improves temporal locality
    # for the repeatedly-read k-th row.
    row_k: DATA_TYPE[n] @ DRAM

    # Main Floyd–Warshall dynamic program.
    #
    # For this benchmark, all edge weights are non-negative (see init_array),
    # so path[k, k] is never negative. Under this condition, the self-updates
    # at (i, k) and (k, j) during the k-th iteration:
    #     path[i, k] = min(path[i, k], path[i, k] + path[k, k])
    #     path[k, j] = min(path[k, j], path[k, k] + path[k, j])
    # are no-ops. Thus, for fixed k, both path[i, k] and path[k, j] are
    # read-only, and it is safe to:
    #   - snapshot the entire k-th row once per k into `row_k`
    #   - reuse path[i, k] across all j for a given (k, i).
    #
    # We also apply simple 2D tiling over (i, j) to improve cache locality.
    for k in seq(0, n):
        # Snapshot the k-th row so that `row_k[j]` can be reused across all i.
        for j in seq(0, n):
            row_k[j] = path[k, j]

        # Tile sizes in i and j. 32 is a reasonable default that aligns with
        # typical cache-line sizes; it can be tuned experimentally.
        #
        # The loop structure after tiling is conceptually:
        #   for I in 0 .. ceil(n / 32):
        #     for i in I-tile:
        #       d_ik = path[i, k]
        #       for J in 0 .. ceil(n / 32):
        #         for j in J-tile:
        #           path[i, j] = min(path[i, j], d_ik + row_k[j])
        #
        # The guards ensure correctness when n is not a multiple of 32.
        for I in seq(0, (n + 31) / 32):
            for i in seq(0, 32):
                if 32 * I + i < n:
                    # Hoist path[i, k] into a scalar reused across all j.
                    d_ik: DATA_TYPE
                    d_ik = path[32 * I + i, k]

                    for J in seq(0, (n + 31) / 32):
                        for j in seq(0, 32):
                            if 32 * J + j < n:
                                path[32 * I + i, 32 * J + j] = min(
                                    path[32 * I + i, 32 * J + j],
                                    d_ik + row_k[32 * J + j],
                                )
EXO END
*/


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

  /* Run Exo kernel. Flatten 2D view to 1D pointer. */
  kernel_floyd_warshall(/*ctxt=*/NULL,
                        n,
                        (DATA_TYPE*)POLYBENCH_ARRAY(path));

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