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


# ---------------------------------------------------------------------------
# Extern for a branch-free minimum on DATA_TYPE
# ---------------------------------------------------------------------------
class _Min(Extern):
    def __init__(self):
        # This is the name that will appear in generated C code.
        super().__init__("min")

    def typecheck(self, args):
        # Two arguments, same type.
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")

        if args[0].type != args[1].type:
            raise _EErr(
                f"expected both arguments to have the same type, "
                f"but got {args[0].type} and {args[1].type}"
            )

        # Return type is the same as the argument type (e.g., DATA_TYPE).
        return args[0].type

    def compile(self, args, prim_type):
        # args[0], args[1] are C strings for the two arguments
        # prim_type is the underlying C primitive type (e.g., float, double, int32_t)
        # Use a ternary operator so we avoid value-dependent control flow in Exo.
        return f"(({prim_type})(({args[0]} < {args[1]}) ? {args[0]} : {args[1]}))"

    def globl(self, prim_type):
        # No additional global C code needed for this extern.
        return ""

    def interpret(self, args):
        # Python-side interpretation, used by Exo for testing and reasoning.
        return args[0] if args[0] < args[1] else args[1]


min = _Min()


# ---------------------------------------------------------------------------
# Baseline Floyd–Warshall kernel in Exo
# ---------------------------------------------------------------------------
@proc
def kernel_floyd_warshall(
    n: size,
    path: DATA_TYPE[n, n] @ DRAM,
):
    # Classic Floyd–Warshall: O(n^3) all-pairs shortest paths.
    # We use the extern `min` to avoid value-dependent branches in Exo.
    for k in seq(0, n):
        for i in seq(0, n):
            for j in seq(0, n):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])


# ---------------------------------------------------------------------------
# Scheduling: improve data locality and reduce redundant memory traffic.
#
# Transformations:
#   1. Hoist path[i, k] out of the innermost j-loop so it is loaded once
#      per (k, i) instead of once per (k, i, j).
#   2. Tile the i and j loops to improve cache locality on large matrices.
#   3. Parallelize the outer i-tile loop using OpenMP pragmas.
#   4. Simplify the IR and rename back to `kernel_floyd_warshall`.
# ---------------------------------------------------------------------------
p = kernel_floyd_warshall

# 1. Hoist `path[i, k]` into a scalar `path_ik` outside the j-loop.
#    This reduces memory traffic for the strided column access path[i, k].
path_ik_exprs = p.find_all("path[i, k]")
p = bind_expr(p, path_ik_exprs, "path_ik")

# 2. Tile the i and j loops.
#    TI and TJ are chosen to fit well in cache on a modern x64 CPU.
TI = 32  # tile size in the i dimension
TJ = 32  # tile size in the j dimension

# Split `i` into (io, ii) where ii iterates within a tile of size TI.
p = divide_loop(p, "i", TI, ("io", "ii"), tail="guard")

# Split `j` into (jo, jj) where jj iterates within a tile of size TJ.
p = divide_loop(p, "j", TJ, ("jo", "jj"), tail="guard")

# 3. Parallelize the outer i-tile loop.
#    For a fixed k, different i-tiles operate on disjoint rows of `path`,
#    so iterations are independent and safe to parallelize.
p = parallelize_loop(p, "io")

# 4. Simplify the resulting code to clean up guards and dead code
#    introduced by tiling and parallelization.
p = simplify(p)

# 5. Ensure the optimized procedure keeps the external name
#    `kernel_floyd_warshall` so the C driver can link against it.
kernel_floyd_warshall = rename(p, "kernel_floyd_warshall")
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