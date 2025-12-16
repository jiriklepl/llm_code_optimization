/**
 * Exo GESUMMV driver: mirrors PolyBench/C gesummv.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"

/* Include the Exo-generated kernel header. */
#include "generated/gesummv/gesummv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
    }
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
from exo.API_scheduling import *  # imported for completeness, even if not used directly
from exo.libs.memories import DRAM

# Tile size along the reduction dimension j.
# Chosen as a small multiple of typical SIMD widths so that the inner
# loop is regular and cache-friendly. This is a compile-time constant.
J_TILE = 64

@proc
def kernel_gesummv(
    n: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    A: DATA_TYPE[n, n] @ DRAM,
    B: DATA_TYPE[n, n] @ DRAM,
    tmp: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
):
    # For each row i, compute two dot products with x:
    #   accA = sum_j A[i, j] * x[j]
    #   accB = sum_j B[i, j] * x[j]
    # then write:
    #   tmp[i] = accA
    #   y[i]   = alpha * accA + beta * accB
    #
    # This is mathematically equivalent to the original kernel:
    #     tmp[i] = 0
    #     y[i]   = 0
    #     for j:
    #         tmp[i] += A[i, j] * x[j]
    #         y[i]   += B[i, j] * x[j]
    #     y[i] = alpha * tmp[i] + beta * y[i]
    #
    # but keeps the partial sums in scalar registers (accA, accB), so the
    # inner loops perform no loads/stores of tmp[i] or y[i]. We also block
    # the j-dimension to improve cache behavior and make the inner loops
    # more vectorization-friendly for the C compiler.

    for i in seq(0, n):
        # Scalar accumulators for this row; these will typically live in registers.
        accA: DATA_TYPE
        accB: DATA_TYPE

        accA = 0.0
        accB = 0.0

        # Main tiled part over j: process full tiles of size J_TILE.
        # The inner loop has constant bounds and unit-stride accesses to
        # A[i, *], B[i, *], and x[*], which the C compiler can readily
        # vectorize (two fused dot-products per row).
        for J in seq(0, n / J_TILE):
            for j in seq(0, J_TILE):
                accA += A[i, J_TILE * J + j] * x[J_TILE * J + j]
                accB += B[i, J_TILE * J + j] * x[J_TILE * J + j]

        # Handle the tail of the row when n is not a multiple of J_TILE.
        # This loop has at most J_TILE iterations per row and preserves
        # the exact same mathematical sum for accA and accB.
        for j in seq(n - n % J_TILE, n):
            accA += A[i, j] * x[j]
            accB += B[i, j] * x[j]

        # Finalize this row: write back the accumulated results once.
        tmp[i] = accA
        y[i] = alpha * accA + beta * accB
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to 1D pointers. */
  kernel_gesummv (/*ctxt=*/NULL, n,
		  (DATA_TYPE*)&alpha,
		  (DATA_TYPE*)&beta,
		  (DATA_TYPE*)POLYBENCH_ARRAY(A),
		  (DATA_TYPE*)POLYBENCH_ARRAY(B),
		  (DATA_TYPE*)POLYBENCH_ARRAY(tmp),
		  (DATA_TYPE*)POLYBENCH_ARRAY(x),
		  (DATA_TYPE*)POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}