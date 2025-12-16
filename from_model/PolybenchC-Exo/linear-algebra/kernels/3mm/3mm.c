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
from exo.API_scheduling import *  # import scheduling API (even if unused directly)
from exo.libs.memories import DRAM

# Tile sizes chosen to balance cache locality and parallelism.
# These can be tuned per target machine.
TILE_I_E = 32
TILE_J_E = 32
TILE_K_E = 32

TILE_J_F = 32
TILE_L_F = 32
TILE_M_F = 32

TILE_I_G = 32
TILE_L_G = 32
TILE_J_G = 32


@proc
def kernel_3mm(
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
    # ------------------------------------------------------------------
    # Phase 1: E := A * B
    #   E[i, j] = sum_k A[i, k] * B[k, j]
    #
    # We use 3D tiling over (i, j, k) to improve cache locality:
    #   - I tiles rows of A and E
    #   - J tiles columns of B and E
    #   - K tiles the reduction dimension over k
    #
    # Each E tile is zeroed once and then accumulated across all K tiles.
    # ------------------------------------------------------------------
    for I in seq(0, (ni + TILE_I_E - 1) / TILE_I_E):
        for J in seq(0, (nj + TILE_J_E - 1) / TILE_J_E):
            # Zero the current E tile.
            for i in seq(0, TILE_I_E):
                if I * TILE_I_E + i < ni:
                    for j in seq(0, TILE_J_E):
                        if J * TILE_J_E + j < nj:
                            E[I * TILE_I_E + i, J * TILE_J_E + j] = 0.0

            # Accumulate contributions from all k-tiles.
            for K in seq(0, (nk + TILE_K_E - 1) / TILE_K_E):
                for i in seq(0, TILE_I_E):
                    if I * TILE_I_E + i < ni:
                        for k in seq(0, TILE_K_E):
                            if K * TILE_K_E + k < nk:
                                a_ik: DATA_TYPE
                                a_ik = A[I * TILE_I_E + i, K * TILE_K_E + k]
                                # Reuse a_ik across the entire J-tile of columns.
                                for j in seq(0, TILE_J_E):
                                    if J * TILE_J_E + j < nj:
                                        E[I * TILE_I_E + i, J * TILE_J_E + j] += \
                                            a_ik * B[K * TILE_K_E + k, J * TILE_J_E + j]

    # ------------------------------------------------------------------
    # Phase 2: F := C * D
    #   F[j, l] = sum_m C[j, m] * D[m, l]
    #
    # We tile over (j, l, m):
    #   - J tiles rows of C and F
    #   - L tiles columns of D and F
    #   - M tiles the reduction dimension over m
    #
    # The l loop (within each tile) is innermost so that F and D are
    # accessed with unit stride.
    # ------------------------------------------------------------------
    for J in seq(0, (nj + TILE_J_F - 1) / TILE_J_F):
        for L in seq(0, (nl + TILE_L_F - 1) / TILE_L_F):
            # Zero the current F tile.
            for j in seq(0, TILE_J_F):
                if J * TILE_J_F + j < nj:
                    for l in seq(0, TILE_L_F):
                        if L * TILE_L_F + l < nl:
                            F[J * TILE_J_F + j, L * TILE_L_F + l] = 0.0

            # Accumulate contributions from all m-tiles.
            for M in seq(0, (nm + TILE_M_F - 1) / TILE_M_F):
                for j in seq(0, TILE_J_F):
                    if J * TILE_J_F + j < nj:
                        for m in seq(0, TILE_M_F):
                            if M * TILE_M_F + m < nm:
                                c_jm: DATA_TYPE
                                c_jm = C[J * TILE_J_F + j, M * TILE_M_F + m]
                                # Reuse c_jm across the L-tile of columns.
                                for l in seq(0, TILE_L_F):
                                    if L * TILE_L_F + l < nl:
                                        F[J * TILE_J_F + j, L * TILE_L_F + l] += \
                                            c_jm * D[M * TILE_M_F + m, L * TILE_L_F + l]

    # ------------------------------------------------------------------
    # Phase 3: G := E * F
    #   G[i, l] = sum_j E[i, j] * F[j, l]
    #
    # We tile over (i, l, j):
    #   - I tiles rows of E and G
    #   - L tiles columns of F and G
    #   - J tiles the reduction dimension over j
    #
    # Each G tile is zeroed once, then updated by all J-tiles, so we
    # exactly reproduce the full sum over j with better locality.
    # ------------------------------------------------------------------
    for I in seq(0, (ni + TILE_I_G - 1) / TILE_I_G):
        for L in seq(0, (nl + TILE_L_G - 1) / TILE_L_G):
            # Zero the current G tile.
            for i in seq(0, TILE_I_G):
                if I * TILE_I_G + i < ni:
                    for l in seq(0, TILE_L_G):
                        if L * TILE_L_G + l < nl:
                            G[I * TILE_I_G + i, L * TILE_L_G + l] = 0.0

            # Accumulate partial sums over j-tiles.
            for J in seq(0, (nj + TILE_J_G - 1) / TILE_J_G):
                for i in seq(0, TILE_I_G):
                    if I * TILE_I_G + i < ni:
                        for j in seq(0, TILE_J_G):
                            if J * TILE_J_G + j < nj:
                                e_ij: DATA_TYPE
                                e_ij = E[I * TILE_I_G + i, J * TILE_J_G + j]
                                # Reuse e_ij across the L-tile of columns.
                                for l in seq(0, TILE_L_G):
                                    if L * TILE_L_G + l < nl:
                                        G[I * TILE_I_G + i, L * TILE_L_G + l] += \
                                            e_ij * F[J * TILE_J_G + j, L * TILE_L_G + l]
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