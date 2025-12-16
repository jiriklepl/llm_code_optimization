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

# Tile sizes chosen for good cache locality on a modern x64 CPU.
# These are compile-time constants for Exo and can be retuned if needed.
TI = 16  # tile size in i
TJ = 16  # tile size in j
TK = 32  # tile size in k (contiguous dimension)


@proc
def kernel_heat_3d_base(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n, n] @ DRAM,
    B: DATA_TYPE[n, n, n] @ DRAM,
):
    # We update only interior points [1, n-1) in each dimension.
    assert n > 2

    # Time-stepping loop (must remain sequential).
    for t in seq(0, tsteps):

        # ------------------------------------------------------------------
        # Phase 1: update B from A using a 7-point stencil.
        #
        # We apply 3D blocking manually using outer tile loops (I, J, K)
        # over the full domain, and inner point loops (ii, jj, kk) over
        # each tile. Guard checks ensure we only update the interior
        # 1 <= i, j, k <= n-2, matching the original kernel.
        # ------------------------------------------------------------------
        for I in seq(0, (n + TI - 1) / TI):
            for J in seq(0, (n + TJ - 1) / TJ):
                for K in seq(0, (n + TK - 1) / TK):

                    # Local scalar reused across stencil evaluations.
                    a_center: DATA_TYPE

                    for ii in seq(0, TI):
                        i = I * TI + ii
                        # Only update interior points; boundaries stay fixed.
                        if i > 0 and i < n - 1:
                            for jj in seq(0, TJ):
                                j = J * TJ + jj
                                if j > 0 and j < n - 1:
                                    for kk in seq(0, TK):
                                        k = K * TK + kk
                                        if k > 0 and k < n - 1:
                                            # Load center once and reuse it.
                                            a_center = A[i, j, k]
                                            B[i, j, k] = (
                                                0.125 * (A[i + 1, j,     k    ] - 2.0 * a_center + A[i - 1, j,     k    ])
                                              + 0.125 * (A[i,     j + 1, k    ] - 2.0 * a_center + A[i,     j - 1, k    ])
                                              + 0.125 * (A[i,     j,     k + 1] - 2.0 * a_center + A[i,     j,     k - 1])
                                              + a_center
                                            )

        # ------------------------------------------------------------------
        # Phase 2: update A from B with the same stencil.
        # Uses the same tiling structure for good cache reuse of B and A.
        # ------------------------------------------------------------------
        for I in seq(0, (n + TI - 1) / TI):
            for J in seq(0, (n + TJ - 1) / TJ):
                for K in seq(0, (n + TK - 1) / TK):

                    b_center: DATA_TYPE

                    for ii in seq(0, TI):
                        i = I * TI + ii
                        if i > 0 and i < n - 1:
                            for jj in seq(0, TJ):
                                j = J * TJ + jj
                                if j > 0 and j < n - 1:
                                    for kk in seq(0, TK):
                                        k = K * TK + kk
                                        if k > 0 and k < n - 1:
                                            b_center = B[i, j, k]
                                            A[i, j, k] = (
                                                0.125 * (B[i + 1, j,     k    ] - 2.0 * b_center + B[i - 1, j,     k    ])
                                              + 0.125 * (B[i,     j + 1, k    ] - 2.0 * b_center + B[i,     j - 1, k    ])
                                              + 0.125 * (B[i,     j,     k + 1] - 2.0 * b_center + B[i,     j,     k - 1])
                                              + b_center
                                            )


# -----------------------------------------------------------------------------
# Scheduling: parallelize tile loops and simplify.
# -----------------------------------------------------------------------------

# Start from the tiled base kernel.
kernel_heat_3d_opt = kernel_heat_3d_base

# Parallelize the outermost I loop of the first (B-from-A) phase.
I0 = kernel_heat_3d_opt.find_loop("I")
kernel_heat_3d_opt = parallelize_loop(kernel_heat_3d_opt, I0)

# Parallelize the outermost I loop of the second (A-from-B) phase.
I1 = kernel_heat_3d_opt.find_loop("I #1")
kernel_heat_3d_opt = parallelize_loop(kernel_heat_3d_opt, I1)

# Let Exo perform algebraic and control-flow simplifications.
kernel_heat_3d_opt = simplify(kernel_heat_3d_opt)

# Export the optimized procedure under the name expected by the C driver.
kernel_heat_3d = rename(kernel_heat_3d_opt, "kernel_heat_3d")
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