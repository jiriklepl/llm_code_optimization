/**
 * Exo 2mm driver: mirrors PolyBench/C 2mm.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"

/* Include the Exo-generated kernel header. */
#include "generated/2mm/2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *  # imported even though we schedule manually
from exo.libs.memories import DRAM

# Tiling factors for cache- and SIMD-friendly blocking.
# These are conservative defaults for a modern x64 core and can be tuned.
TI = 32  # tile size along i (rows of A, tmp, D)
TJ = 32  # tile size along j (cols of tmp, rows of C)
TK = 32  # tile size along k (inner dim NK of A*B)
TL = 32  # tile size along l (cols of C and D)


@proc
def kernel_2mm(
    ni: size,
    nj: size,
    nk: size,
    nl: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    tmp: DATA_TYPE[ni, nj] @ DRAM,
    A: DATA_TYPE[ni, nk] @ DRAM,
    B: DATA_TYPE[nk, nj] @ DRAM,
    C: DATA_TYPE[nj, nl] @ DRAM,
    D: DATA_TYPE[ni, nl] @ DRAM,
):
    # Compute:
    #   D := alpha * A * B * C + beta * D
    #
    # Exactly matching the original two-stage PolyBench kernel:
    #   1) tmp[i,j] = 0;
    #      for k: tmp[i,j] += alpha * A[i,k] * B[k,j];
    #   2) D[i,l] = beta * D[i,l];
    #      for j: D[i,l] += tmp[i,j] * C[j,l];
    #
    # We keep the mathematical semantics identical, but apply
    # cache-aware blocking and loop reordering to improve locality.
    #
    # All data remains in standard C row-major layout; we only
    # change the loop structure.

    # ------------------------------------------------------------
    # Stage 1: tmp = alpha * A * B
    # ------------------------------------------------------------

    # 1a. Initialize tmp to zero (row-major, streaming writes).
    for i in seq(0, ni):
        for j in seq(0, nj):
            tmp[i, j] = 0.0

    # 1b. Blocked accumulation:
    #     for each tile in (i, k, j) space, accumulate contributions
    #     into tmp[i,j] with good locality:
    #
    #     Outer tile loops iterate over:
    #       I  : row tiles of A and tmp
    #       K  : tiles along NK (shared dim of A and B)
    #       J  : column tiles of B and tmp
    #
    #     Inner loops iterate over:
    #       ii : rows within a tile
    #       kk : NK indices within a tile
    #       jj : columns within a tile
    #
    #     Access patterns inside the innermost jj-loop:
    #       - B[global_k, global_j] and tmp[global_i, global_j] are
    #         accessed with unit-stride in j (row-major),
    #       - A[global_i, global_k] is reused across all j in the tile.
    #
    #     For each fixed (i, j), the k-dimension is traversed in
    #     increasing order, just as in the original code. We only
    #     interleave updates across different (i, j) pairs, which
    #     is algebraically safe.

    for I in seq(0, (ni + TI - 1) / TI):
        for K in seq(0, (nk + TK - 1) / TK):
            for J in seq(0, (nj + TJ - 1) / TJ):
                for ii in seq(0, TI):
                    # Global row index.
                    if TI * I + ii < ni:
                        for kk in seq(0, TK):
                            # Global index along NK.
                            if TK * K + kk < nk:
                                # Hoist alpha * A[i,k] out of the inner j-loop
                                # so we pay this multiplication once per (i,k).
                                aik: DATA_TYPE
                                aik = alpha * A[TI * I + ii, TK * K + kk]

                                for jj in seq(0, TJ):
                                    # Global column index into tmp and B.
                                    if TJ * J + jj < nj:
                                        tmp[TI * I + ii, TJ * J + jj] += (
                                            aik * B[TK * K + kk, TJ * J + jj]
                                        )

    # ------------------------------------------------------------
    # Stage 2: D = beta * D + tmp * C
    # ------------------------------------------------------------

    # 2a. Scale D by beta once per element.
    #     This preserves:
    #       D[i,l] = beta * D_in[i,l] + sum_j tmp[i,j] * C[j,l]
    #     while avoiding repeated multiplications by beta inside the
    #     j-reduction.
    for i in seq(0, ni):
        for l in seq(0, nl):
            D[i, l] = beta * D[i, l]

    # 2b. Blocked matrix multiply D += tmp * C.
    #
    # We tile over:
    #   I : rows of tmp and D
    #   J : columns of tmp (shared dim with C)
    #   L : columns of C and D
    #
    # Within a tile, we use the loop order:
    #   ii (rows) -> jj (shared dim) -> ll (cols of C/D)
    #
    # For fixed (i, j) in a tile:
    #   - tmp[i, j] is loaded once into a scalar 't',
    #   - we then stream over l (innermost loop) to update D[i, l]
    #     using C[j, l].
    #
    # This yields unit-stride access to both C[j, l] and D[i, l],
    # allowing the C compiler to vectorize along l.
    #
    # As in stage 1, for each fixed (i, l), the shared j-dimension is
    # visited in strictly increasing order, so the per-element sequence
    # of floating-point operations is preserved.

    for I in seq(0, (ni + TI - 1) / TI):
        for J in seq(0, (nj + TJ - 1) / TJ):
            for L in seq(0, (nl + TL - 1) / TL):
                for ii in seq(0, TI):
                    if TI * I + ii < ni:
                        for jj in seq(0, TJ):
                            if TJ * J + jj < nj:
                                # Cache the tmp[i,j] value in a scalar so it
                                # can be reused across all l in this tile.
                                t: DATA_TYPE
                                t = tmp[TI * I + ii, TJ * J + jj]

                                for ll in seq(0, TL):
                                    if TL * L + ll < nl:
                                        D[TI * I + ii, TL * L + ll] += (
                                            t * C[TJ * J + jj, TL * L + ll]
                                        )
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_2mm (/*ctxt=*/NULL, ni, nj, nk, nl,
	      (DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
	      (DATA_TYPE*)POLYBENCH_ARRAY(tmp),
	      (DATA_TYPE*)POLYBENCH_ARRAY(A),
	      (DATA_TYPE*)POLYBENCH_ARRAY(B),
	      (DATA_TYPE*)POLYBENCH_ARRAY(C),
	      (DATA_TYPE*)POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}