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

@proc
def kernel_heat_3d(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n, n] @ DRAM,
    B: DATA_TYPE[n, n, n] @ DRAM,
):
    assert n > 2

    for t in seq(0, tsteps):
        # Update B from A
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    B[i, j, k] = (
                        0.125 * (A[i + 1, j,     k    ] - 2.0 * A[i, j, k] + A[i - 1, j,     k    ])
                      + 0.125 * (A[i,     j + 1, k    ] - 2.0 * A[i, j, k] + A[i,     j - 1, k    ])
                      + 0.125 * (A[i,     j,     k + 1] - 2.0 * A[i, j, k] + A[i,     j,     k - 1])
                      + A[i, j, k]
                    )

        # Update A from B
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                for k in seq(1, n - 1):
                    A[i, j, k] = (
                        0.125 * (B[i + 1, j,     k    ] - 2.0 * B[i, j, k] + B[i - 1, j,     k    ])
                      + 0.125 * (B[i,     j + 1, k    ] - 2.0 * B[i, j, k] + B[i,     j - 1, k    ])
                      + 0.125 * (B[i,     j,     k + 1] - 2.0 * B[i, j, k] + B[i,     j,     k - 1])
                      + B[i, j, k]
                    )
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