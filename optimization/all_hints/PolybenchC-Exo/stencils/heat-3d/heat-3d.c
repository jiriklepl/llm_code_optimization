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
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# ---------------------------------------------------------------------------
# Base 3D heat stencil kernel (semantics-equivalent to PolyBench heat-3d).
# ---------------------------------------------------------------------------
@proc
def kernel_heat_3d_base(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n, n] @ DRAM,
    B: DATA_TYPE[n, n, n] @ DRAM,
):
    # We update only the interior points [1..n-2] in each dimension.
    assert n > 2

    for t in seq(0, tsteps):
        # ------------------------------------------------------------------
        # First sweep: update B from A.
        # Each update uses a 7-point stencil (center + its 6 axis neighbors).
        # ------------------------------------------------------------------
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    center: DATA_TYPE
                    neighbor_sum: DATA_TYPE

                    center = A[i, j, k]

                    # Sum of the 6 neighbors around the center cell.
                    neighbor_sum = (
                        A[i + 1, j,     k    ] + A[i - 1, j,     k    ] +
                        A[i,     j + 1, k    ] + A[i,     j - 1, k    ] +
                        A[i,     j,     k + 1] + A[i,     j,     k - 1]
                    )

                    # Algebraically equivalent to the original PolyBench update:
                    # B = A
                    #   + 0.125 * [(A_{i+1}+A_{i-1}+A_{j+1}+A_{j-1}+A_{k+1}+A_{k-1})
                    #               - 6*A_center]
                    B[i, j, k] = center + 0.125 * (neighbor_sum - 6.0 * center)

        # ------------------------------------------------------------------
        # Second sweep: update A from B with the same 7-point stencil.
        # ------------------------------------------------------------------
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    center_b: DATA_TYPE
                    neighbor_sum_b: DATA_TYPE

                    center_b = B[i, j, k]

                    neighbor_sum_b = (
                        B[i + 1, j,     k    ] + B[i - 1, j,     k    ] +
                        B[i,     j + 1, k    ] + B[i,     j - 1, k    ] +
                        B[i,     j,     k + 1] + B[i,     j,     k - 1]
                    )

                    A[i, j, k] = center_b + 0.125 * (neighbor_sum_b - 6.0 * center_b)


# ---------------------------------------------------------------------------
# Scheduling / optimization
#
# We apply the following transformations:
#   - Spatial tiling in (i, j) to improve cache locality.
#   - OpenMP-style parallelization of the outer tile loop along i
#     for both sweeps (B <- A and A <- B).
#
# Tile sizes TI and TJ can be tuned for a specific cache hierarchy.
# ---------------------------------------------------------------------------

TI = 16  # tile size in the i (row) dimension
TJ = 16  # tile size in the j (column) dimension


def schedule_kernel_heat_3d():
    p = kernel_heat_3d_base

    # =========================
    # First sweep: B <- A
    # =========================

    # Shift the i- and j-loops so they start at 0 instead of 1.
    # This simplifies subsequent loop division (tiling) while Exo
    # keeps the access pattern semantically identical.
    i0 = p.find_loop("i #0")
    j0 = p.find_loop("j #0")
    p = shift_loop(p, i0, "0")
    p = shift_loop(p, j0, "0")

    # Tile the (i, j) loops of the B <- A sweep.
    # i in [0, n-2)  -> io0 (outer tiles), ii0 (intra-tile)
    i0 = p.find_loop("i #0")
    p = divide_loop(p, i0, TI, ("io0", "ii0"), tail="guard")

    # j in [0, n-2)  -> jo0 (outer tiles), ji0 (intra-tile)
    j0 = p.find_loop("j #0")
    p = divide_loop(p, j0, TJ, ("jo0", "ji0"), tail="guard")

    # Reorder so that tile loops [io0, jo0] are outermost, followed by
    # intra-tile loops [ii0, ji0], then k.
    p = reorder_loops(p, "ii0 jo0")

    # Parallelize across i-tiles for the first sweep.
    # This becomes (by default) an OpenMP parallel for over io0.
    p = parallelize_loop(p, p.find_loop("io0"))

    # =========================
    # Second sweep: A <- B
    # =========================

    # After the first sweep is transformed, the remaining 'i' and 'j'
    # loops belong to the A <- B sweep. Shift them similarly to start
    # from 0 to enable tiling.
    i1 = p.find_loop("i #0")
    j1 = p.find_loop("j #0")
    p = shift_loop(p, i1, "0")
    p = shift_loop(p, j1, "0")

    # Tile the (i, j) loops of the second sweep.
    i1 = p.find_loop("i #0")
    p = divide_loop(p, i1, TI, ("io1", "ii1"), tail="guard")

    j1 = p.find_loop("j #0")
    p = divide_loop(p, j1, TJ, ("jo1", "ji1"), tail="guard")

    # Same loop ordering as in the first sweep.
    p = reorder_loops(p, "ii1 jo1")

    # Parallelize across i-tiles for the second sweep as well.
    p = parallelize_loop(p, p.find_loop("io1"))

    # Simplify algebra and clean up the IR after all transformations.
    p = simplify(p)

    # Export under the name expected by the C driver.
    p = rename(p, "kernel_heat_3d")
    return p


# Final optimized kernel that will be compiled to C.
kernel_heat_3d = schedule_kernel_heat_3d()
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