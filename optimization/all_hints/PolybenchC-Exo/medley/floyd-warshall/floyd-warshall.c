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

# Core Exo imports
from exo import *
from exo.API_scheduling import *  # scheduling primitives
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# ---------------------------------------------------------------------------
# Extern for a generic min() that works on the current DATA_TYPE.
# Implemented as a ternary expression in C to avoid branching on data
# values inside Exo code (which is disallowed).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Baseline Floyd–Warshall kernel in Exo.
#
# Note:
# - DATA_TYPE is a placeholder that will be specialized by the build
#   system (float, double, etc.).
# - path is stored in DRAM with a 2D logical shape [n, n].
# - We use the classic triply-nested Floyd–Warshall structure:
#     for k in 0..n-1
#       for i in 0..n-1
#         for j in 0..n-1
#           path[i,j] = min(path[i,j], path[i,k] + path[k,j])
# ---------------------------------------------------------------------------
@proc
def kernel_floyd_warshall(
    n: size,
    path: DATA_TYPE[n, n] @ DRAM,
):
    for k in seq(0, n):
        for i in seq(0, n):
            for j in seq(0, n):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])


# ---------------------------------------------------------------------------
# Scheduling / optimization section
#
# We apply the following optimizations:
#
# 1. Row staging:
#    - For each k, we stage the k-th row of 'path' (path[k, 0:n]) into a
#      temporary buffer 'row_k'. This improves locality for accesses to
#      path[k, j], which are reused across all i for a fixed (k, j).
#    - The staging transformation preserves semantics by:
#        * Loading the window path[k, 0:n] into row_k before the block.
#        * Rewriting all accesses within the block from path[k, j] to
#          row_k[j].
#        * Storing row_k back to path[k, 0:n] after the block.
#
# 2. Tiling of the i and j loops:
#    - We divide the i and j loops into outer/inner tiles of sizes TI and TJ.
#      This improves cache locality and exposes regular inner loops that
#      the C compiler can better optimize and vectorize.
#    - divide_loop preserves the original lexicographic iteration order
#      and uses guards for non-divisible tails, so semantics are preserved
#      for all n.
#
# 3. Simplification:
#    - Run simplify() to clean up index arithmetic and any redundant code
#      after tiling and staging.
#
# The final optimized kernel is bound back to the name
# 'kernel_floyd_warshall', which is what the C driver calls.
# ---------------------------------------------------------------------------

# Tunable tile sizes for the i and j dimensions.
# These can be adjusted to better fit particular cache hierarchies.
TI = 32
TJ = 32

# Work on a scheduled copy of the original kernel.
kernel_floyd_warshall_opt = kernel_floyd_warshall

# 1) Stage the k-th row of 'path' into a temporary buffer 'row_k'
k_loop = kernel_floyd_warshall_opt.find_loop("k")
k_body = k_loop.body()

# Stage the window path[k, 0:n] around the entire body of the k-loop.
# This creates a temporary buffer:
#   row_k : DATA_TYPE[n]
# and rewrites path[k, j] inside the block to row_k[j], with load/store
# loops inserted before and after the block, respectively.
kernel_floyd_warshall_opt = stage_mem(
    kernel_floyd_warshall_opt,
    k_body,
    "path[k, 0:n]",
    "row_k",
)

# 2) Tile the i loop: i -> (io, ii) with tile size TI
i_loop = kernel_floyd_warshall_opt.find_loop("i")
kernel_floyd_warshall_opt = divide_loop(
    kernel_floyd_warshall_opt,
    i_loop,
    TI,
    ("io", "ii"),
    # tail uses the default "guard" strategy, which safely handles n % TI != 0
)

# 3) Tile the j loop: j -> (jo, ji) with tile size TJ
j_loop = kernel_floyd_warshall_opt.find_loop("j")
kernel_floyd_warshall_opt = divide_loop(
    kernel_floyd_warshall_opt,
    j_loop,
    TJ,
    ("jo", "ji"),
    # Again, the default guarded tail preserves correctness for any n.
)

# 4) Simplify the resulting IR (clean up index arithmetic, dead branches, etc.)
kernel_floyd_warshall_opt = simplify(kernel_floyd_warshall_opt)

# 5) Export the optimized kernel under the original name expected by C.
kernel_floyd_warshall = kernel_floyd_warshall_opt

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