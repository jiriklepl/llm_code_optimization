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
from exo.API_scheduling import *  # import scheduling API (required by guidelines)
from exo.libs.memories import DRAM

# Tile sizes chosen for cache locality on a modern x64 CPU.
# They can be tuned further if desired.
TI = 32  # number of rows per tile
TJ = 32  # number of columns per tile


@proc
def kernel_seidel_2d(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
):
    # We only update the interior 1 .. n-2, so we require at least a 3x3 grid.
    assert n > 2

    # Strength reduction: use a single precomputed reciprocal of 9.0
    # instead of performing a floating-point division in the inner loop.
    inv9: DATA_TYPE
    inv9 = 1.0 / 9.0

    # Time loop (sequential Gauss–Seidel sweeps).
    for t in seq(0, tsteps):
        # Strip-mine the interior rows [1, n-1) into tiles of height TI.
        # Tile index I runs over 0 .. ceil((n-2)/TI) - 1.
        for I in seq(0, (n - 2 + TI - 1) / TI):
            # Within each row tile, iterate over local rows i = 0 .. TI-1.
            for i in seq(0, TI):
                # Global row index of the interior point:
                #   ig = 1 + I*TI + i
                # We guard against going past the last interior row (n-2).
                if 1 + I * TI + i < n - 1:
                    # For this fixed global row, strip-mine columns [1, n-1)
                    # into tiles of width TJ. The order of (ig, jg) is still
                    # lexicographic in (row, column), matching the original code.
                    for J in seq(0, (n - 2 + TJ - 1) / TJ):
                        for j in seq(0, TJ):
                            # Global column index:
                            #   jg = 1 + J*TJ + j
                            if 1 + J * TJ + j < n - 1:
                                # Shorthand for the affine global indices:
                                #   ig = 1 + I*TI + i
                                #   jg = 1 + J*TJ + j
                                #
                                # The 3x3 neighborhood is then:
                                #   rows: ig-1, ig,   ig+1
                                #   cols: jg-1, jg,   jg+1
                                #
                                # which we express directly as affine combinations
                                # of (I, i, J, j). This preserves exactly the same
                                # values as the original kernel.
                                A[1 + I * TI + i, 1 + J * TJ + j] = inv9 * (
                                    A[I * TI + i,     J * TJ + j    ] +
                                    A[I * TI + i,     J * TJ + j + 1] +
                                    A[I * TI + i,     J * TJ + j + 2] +
                                    A[I * TI + i + 1, J * TJ + j    ] +
                                    A[I * TI + i + 1, J * TJ + j + 1] +
                                    A[I * TI + i + 1, J * TJ + j + 2] +
                                    A[I * TI + i + 2, J * TJ + j    ] +
                                    A[I * TI + i + 2, J * TJ + j + 1] +
                                    A[I * TI + i + 2, J * TJ + j + 2]
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