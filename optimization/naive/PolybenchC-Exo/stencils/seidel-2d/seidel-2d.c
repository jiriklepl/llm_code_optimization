/**
 * Exo Seidel-2D driver: mirrors PolyBench/C seidel-2d.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"

/* Include the Exo-generated kernel header. */
#include "generated/seidel-2d/seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM


# Baseline Gauss-Seidel 2D stencil, directly mirroring the PolyBench kernel.
@proc
def kernel_seidel_2d_base(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
):
    # We update the interior region A[1:n-1, 1:n-1], so n must be > 2.
    assert n > 2

    for t in seq(0, tsteps):
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                A[i, j] = (
                    A[i - 1, j - 1] + A[i - 1, j] + A[i - 1, j + 1]
                    + A[i,     j - 1] + A[i,     j] + A[i,     j + 1]
                    + A[i + 1, j - 1] + A[i + 1, j] + A[i + 1, j + 1]
                ) / 9.0


# Start from the baseline kernel and apply scheduling transformations.
kernel_seidel_2d = kernel_seidel_2d_base

# 1. Shift the spatial loops to start from zero.
#    This simplifies later tiling and unrolling while preserving the original
#    lexicographic execution order and the set of updated grid points.
#
#    Original ranges:
#      i in [1, n-1), j in [1, n-1)
#    After shifting both to start at 0:
#      i in [0, n-2), j in [0, n-2)
#    and all accesses are rewritten so that the stencil is now centered
#    at (i+1, j+1) with neighbors in [i .. i+2] x [j .. j+2].
i_loop = kernel_seidel_2d.find_loop("i")
kernel_seidel_2d = shift_loop(kernel_seidel_2d, i_loop, "0")

j_loop = kernel_seidel_2d.find_loop("j")
kernel_seidel_2d = shift_loop(kernel_seidel_2d, j_loop, "0")

# 2. Tile the innermost spatial dimension j by a small constant factor and
#    fully unroll the resulting inner loop.
#
#    We keep the loop nesting order as (t, i, j) logically:
#      - divide_loop on j introduces (jo, ji) with j = 4*jo + ji
#      - the overall visitation order of (i, j) remains row-major
#        (i increasing, then j increasing), so all Gauss-Seidel
#        data dependencies are preserved.
#
#    The tail is handled with a guard, so this works for any n > 2.
j_loop = kernel_seidel_2d.find_loop("j")
kernel_seidel_2d = divide_loop(kernel_seidel_2d, j_loop, 4, ["jo", "ji"], tail="guard")

# 3. Fully unroll the tiny inner ji-loop (extent 4).
#    Unrolling just duplicates the guarded body with ji replaced by
#    constants 0..3, so the execution order is unchanged, but the C
#    compiler sees straight-line code amenable to auto-vectorization.
ji_loop = kernel_seidel_2d.find_loop("ji")
kernel_seidel_2d = unroll_loop(kernel_seidel_2d, ji_loop)

# 4. Clean up the IR (e.g., simplify guards and algebra).
kernel_seidel_2d = simplify(kernel_seidel_2d)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D view to 1D pointer. */
  kernel_seidel_2d(/*ctxt=*/NULL, tsteps, n,
                   (DATA_TYPE*)POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}