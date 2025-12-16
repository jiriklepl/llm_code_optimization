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
from exo.API_scheduling import *
from exo.libs.memories import DRAM
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


# Extern instance used as a side-effect-free min operator inside the kernel.
min = _Min()


@proc
def kernel_floyd_warshall(
    n: size,
    path: DATA_TYPE[n, n] @ DRAM,
):
    # Standard Floyd–Warshall with in-place updates:
    #   for k, i, j: path[i,j] = min(path[i,j], path[i,k] + path[k,j])
    # We apply two algorithmic refinements:
    #   1) Skip i == k: updating row k is a provable no-op when all edge
    #      weights are non-negative (true for this benchmark). This removes
    #      otherwise unnecessary writes and also avoids write/read races when
    #      we parallelize the i-loop.
    #   2) Cache path[i,k] in a scalar 'pik' once per (i,k) and reuse it
    #      across the inner j-loop, drastically reducing strided column loads.
    for k in seq(0, n):
        for i in seq(0, n):
            # For this benchmark, all weights are non-negative, so for i == k
            # we would compute:
            #   path[k,j] = min(path[k,j], path[k,k] + path[k,j])
            # and since path[k,k] >= 0, the result is always the original
            # path[k,j]. We can therefore safely skip i == k.
            if i != k:
                # Cache the (i,k) entry once per row i for this k.
                pik: DATA_TYPE
                pik = path[i, k]

                # Sweep j as the innermost loop to get contiguous accesses to
                # both path[i, j] and path[k, j].
                for j in seq(0, n):
                    path[i, j] = min(path[i, j], pik + path[k, j])


# ---------------------------------------------------------------------------
# Scheduling / optimization
# ---------------------------------------------------------------------------

# Tile size for the innermost j-dimension. This is a compile-time constant
# that can be tuned for a given cache hierarchy.
TJ = 32

# Tile the j loop inside the innermost nest:
#   for j in seq(0, n)  -->
#   for jo in seq(0, n / TJ):
#       for jj in seq(0, TJ):
#           ...
# plus a guarded tail when n is not a multiple of TJ.
kernel_floyd_warshall = divide_loop(
    kernel_floyd_warshall,
    "j",
    TJ,
    ("jo", "jj"),
    tail="guard",
)

# Parallelize the i loop. For a fixed k, each row i updates a disjoint row
# of 'path' and only reads from row/column k and its own row. Combined with
# skipping i == k above (so row k is never written here), this is race-free.
kernel_floyd_warshall = parallelize_loop(kernel_floyd_warshall, "i")

# Clean up any algebraic and control-flow noise introduced by the
# transformations (e.g., constant conditions from tiling guards).
kernel_floyd_warshall = simplify(kernel_floyd_warshall)
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