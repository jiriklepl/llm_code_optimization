/**
 * Exo heat-3d driver: mirrors PolyBench/C heat-3d.c but calls the Exo-generated kernel.
 *
 * Original header:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"

/* Include the Exo-generated kernel header. */
#include "generated/heat-3d/heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
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


# Baseline 7-point 3D heat stencil (algorithmic version).
@proc
def heat_3d_base(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n, n] @ DRAM,
    B: DATA_TYPE[n, n, n] @ DRAM,
):
    # We update interior points [1 : n-1) in each dimension, so n must be > 2.
    assert n > 2

    for t in seq(0, tsteps):
        # Update B from A.
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    B[i, j, k] = (
                        0.125 * (A[i + 1, j,     k    ] - 2.0 * A[i, j, k] + A[i - 1, j,     k    ])
                      + 0.125 * (A[i,     j + 1, k    ] - 2.0 * A[i, j, k] + A[i,     j - 1, k    ])
                      + 0.125 * (A[i,     j,     k + 1] - 2.0 * A[i, j, k] + A[i,     j,     k - 1])
                      + A[i, j, k]
                    )

        # Update A from B.
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    A[i, j, k] = (
                        0.125 * (B[i + 1, j,     k    ] - 2.0 * B[i, j, k] + B[i - 1, j,     k    ])
                      + 0.125 * (B[i,     j + 1, k    ] - 2.0 * B[i, j, k] + B[i,     j - 1, k    ])
                      + 0.125 * (B[i,     j,     k + 1] - 2.0 * B[i, j, k] + B[i,     j,     k - 1])
                      + B[i, j, k]
                    )


# ---------------------------------------------------------------------------
# Scheduling: cache-friendly spatial tiling and OpenMP-style parallelism.
#
# We tile the (i, j) spatial dimensions for both A->B and B->A sweeps to
# improve cache locality. The outer tile loops are then parallelized.
#
# Resulting loop structure inside each time step:
#   for io_* in ...   # outer tiles along i  (parallelized)
#     for jo_* in ... # outer tiles along j
#       for ii_* in ...
#         for ji_* in ...
#           for k in 1..n-1:
#             stencil update
# ---------------------------------------------------------------------------

# Start from the algorithmic version and rename to the externally visible name.
kernel_heat_3d = rename(heat_3d_base, "kernel_heat_3d")

# Tile sizes for i and j. These are modest to keep tiles in cache and
# are safe for all n > 2 (tails are handled by guards).
TI = 4
TJ = 4

# === Tile and parallelize B update (first sweep) ===========================

# 1) Block the i-loop of the B update: i -> (io_B, ii_B).
#    This improves reuse of A's neighboring planes within a tile.
b_i = kernel_heat_3d.find_loop("i")  # First 'i' loop is the B update.
kernel_heat_3d = divide_loop(kernel_heat_3d, b_i, TI, ("io_B", "ii_B"), tail="guard")

# 2) Block the j-loop inside the B update: j -> (jo_B, ji_B).
t_loop = kernel_heat_3d.find_loop("t")
b_io = t_loop.body().find_loop("io_B")
b_ii = b_io.body().find_loop("ii_B")
b_j = b_ii.body().find_loop("j")
kernel_heat_3d = divide_loop(kernel_heat_3d, b_j, TJ, ("jo_B", "ji_B"), tail="guard")

# 3) Reorder ii_B and jo_B so that tile loops (io_B, jo_B) are outermost
#    and inner loops (ii_B, ji_B, k) traverse points contiguously in memory.
t_loop = kernel_heat_3d.find_loop("t")
b_io = t_loop.body().find_loop("io_B")
b_ii = b_io.body().find_loop("ii_B")
kernel_heat_3d = reorder_loops(kernel_heat_3d, b_ii)  # Swap ii_B and the inner jo_B.

# 4) Parallelize the outermost spatial tile loop over io_B.
#    Exo will emit an OpenMP parallel-for by default.
kernel_heat_3d = parallelize_loop(kernel_heat_3d, "io_B")

# === Tile and parallelize A update (second sweep) ==========================

# 5) Block the i-loop of the A update (the remaining 'i' loop): i -> (io_A, ii_A).
a_i = kernel_heat_3d.find_loop("i")  # After step 1, this 'i' is the A update.
kernel_heat_3d = divide_loop(kernel_heat_3d, a_i, TI, ("io_A", "ii_A"), tail="guard")

# 6) Block the j-loop inside the A update: j -> (jo_A, ji_A).
t_loop = kernel_heat_3d.find_loop("t")
a_io = t_loop.body().find_loop("io_A")
a_ii = a_io.body().find_loop("ii_A")
a_j = a_ii.body().find_loop("j")
kernel_heat_3d = divide_loop(kernel_heat_3d, a_j, TJ, ("jo_A", "ji_A"), tail="guard")

# 7) Reorder ii_A and jo_A so that tile loops (io_A, jo_A) are outermost.
t_loop = kernel_heat_3d.find_loop("t")
a_io = t_loop.body().find_loop("io_A")
a_ii = a_io.body().find_loop("ii_A")
kernel_heat_3d = reorder_loops(kernel_heat_3d, a_ii)

# 8) Parallelize the outermost spatial tile loop over io_A.
kernel_heat_3d = parallelize_loop(kernel_heat_3d, "io_A")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 3D views to 1D pointers. */
  kernel_heat_3d(/*ctxt=*/NULL, tsteps, n,
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

  return 0;
}