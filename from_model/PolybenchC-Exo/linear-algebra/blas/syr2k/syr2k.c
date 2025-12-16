/**
 * Exo SYR2K driver: mirrors PolyBench/C syr2k.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"

/* Include the Exo-generated kernel header. */
#include "generated/syr2k/syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *  # imported for completeness; not used directly
from exo.libs.memories import DRAM

# Tile sizes chosen to improve cache locality on a modern x64 CPU.
# They can be tuned per-target; they must be positive compile-time constants.
TI = 32
TJ = 32
TK = 32

@proc
def kernel_syr2k(
    n: size,
    m: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[n, n] @ DRAM,
    A: DATA_TYPE[n, m] @ DRAM,
    B: DATA_TYPE[n, m] @ DRAM,
):
    # 3D tiled symmetric rank-2k update on the lower triangle of C.
    #
    # We work on rectangular tiles in (i, j) and blocks of the reduction
    # dimension k. Within each tile element (i, j) we keep a scalar
    # accumulator in a register:
    #
    #   acc = beta * C[i, j]
    #   for k: acc += alpha * (A[j, k] * B[i, k] + B[j, k] * A[i, k])
    #   C[i, j] = acc
    #
    # This fuses the original "scale by beta" and "rank-2k update" loops,
    # reduces C traffic to one load + one store per (i, j), and streams A
    # and B contiguously along k (row-major layout).
    for I in seq(0, (n + TI - 1) / TI):
        for J in seq(0, (n + TJ - 1) / TJ):
            for i0 in seq(0, TI):
                # Global row index for this tile
                if TI * I + i0 < n:
                    for j0 in seq(0, TJ):
                        # Global column index for this tile
                        if TJ * J + j0 < n:
                            # Only update the lower triangular part: j <= i.
                            # Tiles that lie entirely above the diagonal are
                            # skipped by this guard.
                            if TJ * J + j0 <= TI * I + i0:
                                # Scalar accumulator kept in a register across
                                # the k-reduction for this (i, j).
                                acc: DATA_TYPE
                                acc = beta * C[TI * I + i0, TJ * J + j0]

                                # Tile the reduction dimension k to improve
                                # cache reuse of the rows of A and B. For each
                                # (i, j) pair we walk k contiguously, which is
                                # friendly to the hardware prefetcher and
                                # enables SIMD auto-vectorization.
                                for K in seq(0, (m + TK - 1) / TK):
                                    for k0 in seq(0, TK):
                                        if TK * K + k0 < m:
                                            acc += alpha * (
                                                A[TJ * J + j0, TK * K + k0]
                                                * B[TI * I + i0, TK * K + k0]
                                                + B[TJ * J + j0, TK * K + k0]
                                                * A[TI * I + i0, TK * K + k0]
                                            )

                                # Write back the updated C element once.
                                C[TI * I + i0, TJ * J + j0] = acc
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_syr2k (/*ctxt=*/NULL, n, m,
		(DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
		(DATA_TYPE*)POLYBENCH_ARRAY(C),
		(DATA_TYPE*)POLYBENCH_ARRAY(A),
		(DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}