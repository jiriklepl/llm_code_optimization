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
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Tunable tile sizes for the innermost output dimensions.
# These are compile-time constants; they can be adjusted to better fit
# the target machine's cache hierarchy.
TILE_J = 32  # tile size along nj (columns of tmp and B)
TILE_L = 32  # tile size along nl (columns of D and C)


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
    # Compute: D := alpha * A * B * C + beta * D

    # 1. Initialize tmp to zero.
    for i in seq(0, ni):
        for j in seq(0, nj):
            tmp[i, j] = 0.0

    # 2. Compute tmp = alpha * A * B.
    #    Loop order (i, k, j) gives:
    #      - unit-stride access to B[k, j] along j,
    #      - unit-stride access to tmp[i, j] along j,
    #      - reuse of alpha * A[i, k] across the j-loop.
    for i in seq(0, ni):
        for k in seq(0, nk):
            akk: DATA_TYPE
            akk = alpha * A[i, k]
            for j in seq(0, nj):
                tmp[i, j] += akk * B[k, j]

    # 3. Scale D by beta.
    for i in seq(0, ni):
        for j in seq(0, nl):
            D[i, j] = beta * D[i, j]

    # 4. Accumulate tmp * C into D.
    #    Loop order (i, k, j) gives:
    #      - unit-stride access to C[k, j] along j,
    #      - unit-stride access to D[i, j] along j,
    #      - reuse of tmp[i, k] across the j-loop.
    for i in seq(0, ni):
        for k in seq(0, nj):
            tik: DATA_TYPE
            tik = tmp[i, k]
            for j in seq(0, nl):
                D[i, j] += tik * C[k, j]


# Apply scheduling transformations to improve data locality and expose parallelism.
def schedule_kernel_2mm(p):
    # Tile the innermost j loop of the A*B phase (tmp update).
    k_ab = p.find_loop("k")               # k loop in phase 2
    j_ab = k_ab.body().find_loop("j")     # inner j loop within that k loop
    p = divide_loop(p, j_ab, TILE_J, ("jo", "ji"))

    # Parallelize over i in the A*B phase.
    i_ab = p.find_loop("i #1")            # second top-level i loop (phase 2)
    p = parallelize_loop(p, i_ab)

    # Tile the innermost j loop of the tmp*C phase (D update).
    k_dc = p.find_loop("k #1")            # k loop in phase 4
    j_dc = k_dc.body().find_loop("j")     # inner j loop within that k loop
    p = divide_loop(p, j_dc, TILE_L, ("jo", "ji"))

    # Parallelize over i in the D scaling and D update phases.
    i_scale = p.find_loop("i #2")         # third top-level i loop (scale D)
    p = parallelize_loop(p, i_scale)
    i_dc = p.find_loop("i #3")            # fourth top-level i loop (D update)
    p = parallelize_loop(p, i_dc)

    return p


kernel_2mm = schedule_kernel_2mm(kernel_2mm)
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