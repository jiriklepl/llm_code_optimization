/**
 * Exo 3mm driver: mirrors PolyBench/C 3mm.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"

/* Include the Exo-generated kernel header. */
#include "generated/3mm/3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Base triple-matrix-multiply kernel.
# We structure each GEMM as:
#   1) a separate zero-initialization pass, and
#   2) an i-k-j "row-panel" multiply.
# This gives good locality:
#   - A and C are streamed row-wise (contiguous in k),
#   - B, D, and F are streamed along their column index j
#     (inner-most loop), improving cache reuse and vectorization.
@proc
def kernel_3mm_base(
    ni: size,
    nj: size,
    nk: size,
    nl: size,
    nm: size,
    E: DATA_TYPE[ni, nj] @ DRAM,
    A: DATA_TYPE[ni, nk] @ DRAM,
    B: DATA_TYPE[nk, nj] @ DRAM,
    F: DATA_TYPE[nj, nl] @ DRAM,
    C: DATA_TYPE[nj, nm] @ DRAM,
    D: DATA_TYPE[nm, nl] @ DRAM,
    G: DATA_TYPE[ni, nl] @ DRAM,
):
    # Basic shape sanity; PolyBench always uses positive sizes.
    assert ni > 0
    assert nj > 0
    assert nk > 0
    assert nl > 0
    assert nm > 0

    # ------------------------------------------------------------------
    # Phase 1: E = A * B
    # ------------------------------------------------------------------

    # Zero-initialize E.
    for iE in seq(0, ni):
        for jE in seq(0, nj):
            E[iE, jE] = 0.0

    # Row-panel GEMM: E[iE, jE] += A[iE, kE] * B[kE, jE]
    #   iE: row of A/E
    #   kE: inner dimension, iterating along A's contiguous dimension
    #   jE: column of B/E, kept innermost for good locality in B/E
    for iE in seq(0, ni):
        for kE in seq(0, nk):
            for jE in seq(0, nj):
                E[iE, jE] += A[iE, kE] * B[kE, jE]

    # ------------------------------------------------------------------
    # Phase 2: F = C * D
    # ------------------------------------------------------------------

    # Zero-initialize F.
    for iF in seq(0, nj):
        for jF in seq(0, nl):
            F[iF, jF] = 0.0

    # Row-panel GEMM: F[iF, jF] += C[iF, kF] * D[kF, jF]
    #   iF: row of C/F
    #   kF: inner dimension, iterating along C
    #   jF: column of D/F, kept innermost for locality
    for iF in seq(0, nj):
        for kF in seq(0, nm):
            for jF in seq(0, nl):
                F[iF, jF] += C[iF, kF] * D[kF, jF]

    # ------------------------------------------------------------------
    # Phase 3: G = E * F
    # ------------------------------------------------------------------

    # Zero-initialize G.
    for iG in seq(0, ni):
        for jG in seq(0, nl):
            G[iG, jG] = 0.0

    # Row-panel GEMM: G[iG, jG] += E[iG, kG] * F[kG, jG]
    #   iG: row of E/G
    #   kG: inner dimension
    #   jG: column of F/G, innermost for locality
    for iG in seq(0, ni):
        for kG in seq(0, nj):
            for jG in seq(0, nl):
                G[iG, jG] += E[iG, kG] * F[kG, jG]


# ----------------------------------------------------------------------
# Scheduling: cache-friendly tiling + OpenMP-style parallelization
# ----------------------------------------------------------------------

p = kernel_3mm_base

# Column-block size. 32 is a good generic choice on modern x86:
# - large enough to amortize loop overhead,
# - small enough to fit in L1/L2 when combined with several rows.
tile_j = 32

# Tile the column loops of the three GEMM accumulation phases.
# We only tile the j-dimension (column) which is fully independent
# across iterations, so the transformation is equivalence-preserving.

# Tile jE in the E = A * B accumulation loop (second jE loop).
p = divide_loop(
    p,
    p.find_loop("jE #1"),       # second jE loop: accumulation
    tile_j,
    ("jE_outer", "jE_inner"),
    tail="guard",               # guarded tail handles non-multiple sizes
)

# Tile jF in the F = C * D accumulation loop (second jF loop).
p = divide_loop(
    p,
    p.find_loop("jF #1"),
    tile_j,
    ("jF_outer", "jF_inner"),
    tail="guard",
)

# Tile jG in the G = E * F accumulation loop (second jG loop).
p = divide_loop(
    p,
    p.find_loop("jG #1"),
    tile_j,
    ("jG_outer", "jG_inner"),
    tail="guard",
)

# Parallelize the outer row loops of each GEMM accumulation.
# Rows are independent: each thread works on disjoint slices of E, F, or G.

p = parallelize_loop(p, p.find_loop("iE #1"))  # E accumulation
p = parallelize_loop(p, p.find_loop("iF #1"))  # F accumulation
p = parallelize_loop(p, p.find_loop("iG #1"))  # G accumulation

# Export the scheduled kernel under the name expected by the C driver.
kernel_3mm = rename(p, "kernel_3mm")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_3mm (/*ctxt=*/NULL, ni, nj, nk, nl, nm,
              (DATA_TYPE*)POLYBENCH_ARRAY(E),
              (DATA_TYPE*)POLYBENCH_ARRAY(A),
              (DATA_TYPE*)POLYBENCH_ARRAY(B),
              (DATA_TYPE*)POLYBENCH_ARRAY(F),
              (DATA_TYPE*)POLYBENCH_ARRAY(C),
              (DATA_TYPE*)POLYBENCH_ARRAY(D),
              (DATA_TYPE*)POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}