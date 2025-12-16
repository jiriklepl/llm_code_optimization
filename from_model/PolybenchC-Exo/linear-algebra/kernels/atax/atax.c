--- START OF SOURCE: atax.c ---
/**
 * Exo ATAX driver: mirrors PolyBench/C atax.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"

/* Include the Exo-generated kernel header. */
#include "generated/atax/atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
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
from exo.API_scheduling import *  # imported for completeness (no direct use here)
from exo.libs.memories import DRAM

# Tile sizes chosen to improve cache reuse and enable efficient SIMD codegen.
# - TJ controls how many columns (entries of x and y) are processed together.
#   A tile of size TJ touches a TJ-wide contiguous slice of each row of A and
#   the matching slice of x or y, which keeps those cache lines hot.
TJ = 64

@proc
def kernel_atax(
    m: size,
    n: size,
    A: DATA_TYPE[m, n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
    tmp: DATA_TYPE[m] @ DRAM,
):
    # Phase 0: initialize y to zero.
    #
    # This matches the original kernel, where y is cleared before it
    # accumulates the contributions A[i, j] * tmp[i].
    for j in seq(0, n):
        y[j] = 0.0

    # Phase 1: tmp = A * x
    # ---------------------
    # For each row i of A, compute one dot product between A[i, :] and x[:]:
    #
    #   tmp[i] = sum_{j=0..n-1} A[i, j] * x[j]
    #
    # The inner loops over j are tiled by TJ so that we work on contiguous
    # blocks of A[i, j] and x[j]. The order of j (0..n-1) is preserved, so
    # floating-point rounding behavior for each tmp[i] is identical to the
    # simple, untiled version.
    for i in seq(0, m):
        tmp[i] = 0.0
        # Outer loop over column tiles.
        for J in seq(0, (n + TJ - 1) / TJ):
            # Inner loop over elements within a tile.
            for j in seq(0, TJ):
                # Guard handles the last (possibly partial) tile without
                # accessing out-of-bounds elements.
                if J * TJ + j < n:
                    tmp[i] += A[i, J * TJ + j] * x[J * TJ + j]

    # Phase 2: y = A^T * tmp
    # -----------------------
    # Mathematically:
    #
    #   y[j] = sum_{i=0..m-1} A[i, j] * tmp[i]
    #
    # Starting from y ≡ 0, this is equivalent to the original Exo kernel's:
    #
    #   for i:
    #       ... compute tmp[i] ...
    #       for j:
    #           y[j] += A[i, j] * tmp[i]
    #
    # We reorganize the loops to tile the output dimension j, so that each
    # tile of y (and the matching slice of each row of A) is processed
    # completely before moving on to the next tile. This greatly improves
    # temporal locality for y and enables efficient SIMD along j.
    #
    # Importantly, for each fixed j, the contributions from i = 0..m-1 are
    # still applied in the same order as in the original kernel, so the
    # floating-point reduction order per y[j] is unchanged.
    for J in seq(0, (n + TJ - 1) / TJ):
        # For each row i, apply its contribution to the current y-tile.
        for i in seq(0, m):
            # Hold tmp[i] in a scalar so the compiler can keep it in a register
            # across the inner loop over j, reducing loads from memory.
            ti: DATA_TYPE
            ti = tmp[i]
            for j in seq(0, TJ):
                if J * TJ + j < n:
                    y[J * TJ + j] += A[i, J * TJ + j] * ti
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten views to raw pointers. */
  kernel_atax (/*ctxt=*/NULL, m, n,
               (DATA_TYPE*)POLYBENCH_ARRAY(A),
               (DATA_TYPE*)POLYBENCH_ARRAY(x),
               (DATA_TYPE*)POLYBENCH_ARRAY(y),
               (DATA_TYPE*)POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}
--- END OF SOURCE: atax.c ---