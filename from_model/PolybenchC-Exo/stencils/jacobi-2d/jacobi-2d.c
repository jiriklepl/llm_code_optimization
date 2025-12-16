/**
 * Exo Jacobi-2D driver: mirrors PolyBench/C jacobi-2d.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Include the Exo-generated kernel header. */
#include "generated/jacobi-2d/jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
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

# Tunable spatial tile sizes. These are chosen to give good cache locality on
# typical x64 machines while keeping the code simple. They can be adjusted and
# the kernel recompiled without changing the C driver.
TI = 32
TJ = 32


@proc
def kernel_jacobi_2d(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
    B: DATA_TYPE[n, n] @ DRAM,
):
    # We update only interior points [1 : n-1), so n must be > 2.
    assert n > 2

    # Local index scalars used when reconstructing global indices from tiles.
    i: size
    j: size

    # Time loop is kept outermost and strictly sequential, as each step depends
    # on the result of the previous one.
    for t in seq(0, tsteps):
        # ------------------------------------------------------------------
        # Phase 1: update B from A using a 5‑point stencil.
        #
        # The interior region is (n-2) × (n-2) points with i,j ∈ [1, n-1).
        # We tile this interior domain into TI×TJ blocks using 0-based tile
        # indices (Ii, Jj). For each tile we iterate over local coordinates
        # (ii, jj) and reconstruct the global indices:
        #   i = 1 + Ii * TI + ii
        #   j = 1 + Jj * TJ + jj
        # Guards ensure we skip iterations that would fall outside [1, n-1).
        # This preserves the original computation exactly, while improving
        # cache locality.
        # ------------------------------------------------------------------
        for Ii in seq(0, (n - 2 + TI - 1) // TI):
            for Jj in seq(0, (n - 2 + TJ - 1) // TJ):
                for ii in seq(0, TI):
                    # Global row index; valid interior rows satisfy 1 <= i < n-1.
                    i = 1 + Ii * TI + ii
                    if i < n - 1:
                        for jj in seq(0, TJ):
                            # Global column index; valid interior cols satisfy 1 <= j < n-1.
                            j = 1 + Jj * TJ + jj
                            if j < n - 1:
                                B[i, j] = 0.2 * (
                                    A[i, j]
                                    + A[i, j - 1]
                                    + A[i, j + 1]
                                    + A[i + 1, j]
                                    + A[i - 1, j]
                                )

        # ------------------------------------------------------------------
        # Phase 2: update A from B using the same 5‑point stencil.
        #
        # We apply the same tiling pattern to the A update. Within this phase
        # all updates read only from B (which is not modified in this phase),
        # so the reordering implied by tiling is semantics‑preserving.
        # ------------------------------------------------------------------
        for Ii in seq(0, (n - 2 + TI - 1) // TI):
            for Jj in seq(0, (n - 2 + TJ - 1) // TJ):
                for ii in seq(0, TI):
                    i = 1 + Ii * TI + ii
                    if i < n - 1:
                        for jj in seq(0, TJ):
                            j = 1 + Jj * TJ + jj
                            if j < n - 1:
                                A[i, j] = 0.2 * (
                                    B[i, j]
                                    + B[i, j - 1]
                                    + B[i, j + 1]
                                    + B[i + 1, j]
                                    + B[i - 1, j]
                                )
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_jacobi_2d(/*ctxt=*/NULL,
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