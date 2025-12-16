/**
 * Exo DURBIN driver: mirrors PolyBench/C durbin.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"

/* Include the Exo-generated kernel header. */
#include "generated/durbin/durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Baseline Durbin kernel in Exo.
# DATA_TYPE is a placeholder that will be substituted with the concrete
# scalar type (e.g., float or double) during code generation.
@proc
def kernel_durbin(
    n: size,
    r: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    z: DATA_TYPE[n] @ DRAM
    alpha: DATA_TYPE
    beta: DATA_TYPE
    sum: DATA_TYPE

    # Problem size must be positive for the recursion to make sense.
    assert n > 0

    # Initial conditions (k = 0 stage).
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]

    # Main Durbin recursion.
    for k in seq(1, n):
        # Update beta based on previous reflection coefficient alpha.
        beta = (1.0 - alpha * alpha) * beta

        # Compute sum = Σ_{i=0}^{k-1} r[i] * y[k-1-i].
        # This reindexing gives forward, contiguous access to r.
        sum = 0.0
        for i in seq(0, k):
            sum += r[i] * y[k - i - 1]

        # New reflection coefficient.
        alpha = -(r[k] + sum) / beta

        # Update the solution into the scratch buffer z.
        for i in seq(0, k):
            z[i] = y[i] + alpha * y[k - i - 1]

        # Copy updated values back into y.
        for i in seq(0, k):
            y[i] = z[i]

        # Set the newest element.
        y[k] = alpha


# ---------------------------------------------------------------------------
# Schedule / optimization
# ---------------------------------------------------------------------------
# We optimize the three inner i-loops:
#  1) the dot-product accumulating into sum
#  2) the update into z
#  3) the copy-back from z to y
#
# For each of them we:
#   - strip-mine with a factor of 4
#   - fully unroll the inner (factor-4) loop
#
# This improves instruction-level parallelism and helps the C compiler
# generate efficient code while correctly handling non-multiple-of-4 sizes
# via guarded remainder loops.

unroll_factor = 4

# Grab cursors to the three original i-loops before we start rewriting.
i_loop_sum = kernel_durbin.find("for i in _:_")         # first i-loop
i_loop_z   = kernel_durbin.find("for i in _:_ #1")      # second i-loop
i_loop_y   = kernel_durbin.find("for i in _:_ #2")      # third i-loop

# 1) Strip-mine and unroll the sum accumulation loop.
kernel_durbin = divide_loop(
    kernel_durbin,
    i_loop_sum,
    unroll_factor,
    ("io0", "ii0"),
    tail="guard",
)
kernel_durbin = unroll_loop(kernel_durbin, "ii0")

# 2) Strip-mine and unroll the z update loop.
kernel_durbin = divide_loop(
    kernel_durbin,
    i_loop_z,
    unroll_factor,
    ("io1", "ii1"),
    tail="guard",
)
kernel_durbin = unroll_loop(kernel_durbin, "ii1")

# 3) Strip-mine and unroll the y copy-back loop.
kernel_durbin = divide_loop(
    kernel_durbin,
    i_loop_y,
    unroll_factor,
    ("io2", "ii2"),
    tail="guard",
)
kernel_durbin = unroll_loop(kernel_durbin, "ii2")

# Clean up expressions and eliminate any trivial guards or dead code
# introduced by the strip-mining and unrolling.
kernel_durbin = simplify(kernel_durbin)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 1D views to raw pointers. */
  kernel_durbin(/*ctxt=*/NULL,
                n,
                (DATA_TYPE*)POLYBENCH_ARRAY(r),
                (DATA_TYPE*)POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}