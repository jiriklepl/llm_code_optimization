/**
 * Exo Jacobi-1D driver: mirrors PolyBench/C jacobi-1d.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"

/* Include the Exo-generated kernel header. */
#include "generated/jacobi-1d/jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
    {
      A[i] = ((DATA_TYPE) i + 2) / n;
      B[i] = ((DATA_TYPE) i + 3) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
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

# Spatial tile size along the 1D domain. This is a tuning knob; 1024 is
# a reasonable default for modern x64 CPUs with 32 KiB L1 data caches.
I_TILE: int = 1024


@proc
def kernel_jacobi_1d(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n] @ DRAM,
    B: DATA_TYPE[n] @ DRAM,
):
    # The stencil touches A[i-1] and A[i+1] / B[i-1] and B[i+1] for interior i,
    # so we require at least 3 points.
    assert n > 2

    # Two-sweep Jacobi update: for each time step t,
    #   1) B := stencil(A)
    #   2) A := stencil(B)
    #
    # To make spatial tiling easier, we use a 0-based loop over i in [0, n-2)
    # and shift all array accesses by +1. This is algebraically equivalent to
    # the original loops over i in [1, n-1).
    for t in seq(0, tsteps):
        # Phase 1: update B from A (A is read-only in this phase).
        for i in seq(0, n - 2):
            B[i + 1] = 0.33333 * (A[i] + A[i + 1] + A[i + 2])

        # Phase 2: update A from B (B is read-only in this phase).
        for i in seq(0, n - 2):
            A[i + 1] = 0.33333 * (B[i] + B[i + 1] + B[i + 2])


# ---------------------------------------------------------------------------
# Scheduling: cache-friendly tiling in the spatial dimension and
#             coarse-grain OpenMP parallelism across tiles.
#
# We keep the outer time loop t sequential to preserve the Jacobi
# recurrence across time steps. Within a single time step:
#   - Phase 1 reads A and writes B[1..n-2].
#   - Phase 2 reads B and writes A[1..n-2].
# Different spatial indices i are independent within each phase, so we
# can safely tile and parallelize over i.
# ---------------------------------------------------------------------------

# Tile the spatial loops in both phases by I_TILE. The loops operate over
# the range i in [0, n-2), so tiling splits that range into chunks of size
# at most I_TILE. This improves data locality for A and B and creates
# a natural unit of work for each thread.
kernel_jacobi_1d = divide_loop(
    kernel_jacobi_1d,
    "i",                       # the i-loop in the B-update phase
    I_TILE,
    ("i_outer_B", "i_inner_B"),
    tail="guard",
)

kernel_jacobi_1d = divide_loop(
    kernel_jacobi_1d,
    "i #1",                    # the i-loop in the A-update phase
    I_TILE,
    ("i_outer_A", "i_inner_A"),
    tail="guard",
)

# Parallelize across tiles in each phase. For a fixed time step t:
#   - In the B phase each iteration writes a distinct B[i+1] and only
#     reads A, which is read-only, so tiles are independent.
#   - In the A phase each iteration writes a distinct A[i+1] and only
#     reads B, which is read-only, so tiles are independent.
# The two phases remain sequential in t, preserving Jacobi semantics.
outer_B = kernel_jacobi_1d.find_loop("i_outer_B")
outer_A = kernel_jacobi_1d.find_loop("i_outer_A")

kernel_jacobi_1d = parallelize_loop(kernel_jacobi_1d, outer_B)
kernel_jacobi_1d = parallelize_loop(kernel_jacobi_1d, outer_A)

# Clean up the IR after transformations (e.g., simplify guards/tails).
kernel_jacobi_1d = simplify(kernel_jacobi_1d)

EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 1D views to 1D pointers. */
  kernel_jacobi_1d (/*ctxt=*/NULL,
                    tsteps, n,
                    (DATA_TYPE*)POLYBENCH_ARRAY(A),
                    (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}