/**
 * Exo TRISOLV driver: mirrors PolyBench/C trisolv.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

/* Include the Exo-generated kernel header. */
#include "generated/trisolv/trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *  # imported for completeness, even though we hand-schedule below
from exo.libs.memories import DRAM

# Tuning parameters for the inner j-loop.
# TILE_J controls how many previous entries of x are processed per tile.
# UNROLL_J is a manual unroll factor inside each tile and must divide TILE_J.
TILE_J = 32
UNROLL_J = 4

@proc
def kernel_trisolv(
    n: size,
    L: DATA_TYPE[n, n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    b: DATA_TYPE[n] @ DRAM,
):
    # Forward substitution solve for the lower-triangular system L x = b.
    # The i-loop must remain sequential because x[i] depends on x[0..i-1].
    for i in seq(0, n):
        # Keep the current solution component in a scalar accumulator.
        # This removes repeated loads/stores of x[i] in the inner loop.
        s: DATA_TYPE
        s = b[i]

        # Process full tiles of size TILE_J with a manually unrolled inner loop.
        # For a given i, the valid reduction range is j in [0, i).
        # We cover the prefix [0, i) as:
        #   - floor(i / TILE_J) full tiles of size TILE_J, each unrolled by UNROLL_J
        #   - a scalar tail from floor(i / TILE_J) * TILE_J up to i-1
        for J in seq(0, i / TILE_J):
            for jj in seq(0, TILE_J / UNROLL_J):
                # Unrolled updates over four consecutive j positions.
                s = s - L[i, J * TILE_J + jj * UNROLL_J] * x[J * TILE_J + jj * UNROLL_J]
                s = s - L[i, J * TILE_J + jj * UNROLL_J + 1] * x[J * TILE_J + jj * UNROLL_J + 1]
                s = s - L[i, J * TILE_J + jj * UNROLL_J + 2] * x[J * TILE_J + jj * UNROLL_J + 2]
                s = s - L[i, J * TILE_J + jj * UNROLL_J + 3] * x[J * TILE_J + jj * UNROLL_J + 3]

        # Handle any remaining elements in the last (possibly partial) tile.
        for j in seq(i / TILE_J * TILE_J, i):
            s = s - L[i, j] * x[j]

        # Final normalization by the diagonal entry of L.
        x[i] = s / L[i, i]
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten views to raw pointers. */
  kernel_trisolv (/*ctxt=*/NULL,
                  n,
                  (DATA_TYPE*)POLYBENCH_ARRAY(L),
                  (DATA_TYPE*)POLYBENCH_ARRAY(x),
                  (DATA_TYPE*)POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}