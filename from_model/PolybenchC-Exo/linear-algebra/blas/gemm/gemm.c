/**
 * Exo GEMM driver: mirrors PolyBench/C gemm.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

/* Include the Exo-generated kernel header. */
#include "generated/gemm/gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Tile sizes chosen for cache locality on a modern x64 CPU.
# They control the height/width/depth of the tiles of C/A/B that are
# operated on while resident in cache.
TI: int = 64
TJ: int = 64
TK: int = 64

@proc
def kernel_gemm_tiled(
    ni: size,
    nj: size,
    nk: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[ni, nj] @ DRAM,
    A: DATA_TYPE[ni, nk] @ DRAM,
    B: DATA_TYPE[nk, nj] @ DRAM,
):
    # 2D tiling of the (i,j) space of C. Each (ii,jj) tile covers
    # a TI x TJ submatrix of C (plus edge handling for non-multiples).
    for ii in seq(0, (ni + TI - 1) / TI):
        for jj in seq(0, (nj + TJ - 1) / TJ):

            # Scale the current C tile by beta once, before accumulating A*B.
            # This preserves the original semantics:
            #   C[i,j] = beta * C[i,j]
            #   C[i,j] += alpha * sum_k A[i,k] * B[k,j]
            # but avoids a separate full pass over C.
            for i in seq(0, TI):
                if ii * TI + i < ni:
                    for j in seq(0, TJ):
                        if jj * TJ + j < nj:
                            C[ii * TI + i, jj * TJ + j] = beta * C[ii * TI + i, jj * TJ + j]

            # Sweep over tiles in the reduction dimension k.
            # For each C tile (ii,jj) we iterate over all K tiles and
            # accumulate alpha * A * B into C.
            for kk in seq(0, (nk + TK - 1) / TK):
                for i in seq(0, TI):
                    if ii * TI + i < ni:
                        for k in seq(0, TK):
                            if kk * TK + k < nk:
                                # Hoist alpha * A[i,k] out of the inner j-loop.
                                # This reduces one multiplication per inner iteration:
                                #   C[i,j] += (alpha*A[i,k]) * B[k,j]
                                a_val: DATA_TYPE
                                a_val = alpha * A[ii * TI + i, kk * TK + k]
                                for j in seq(0, TJ):
                                    if jj * TJ + j < nj:
                                        C[ii * TI + i, jj * TJ + j] += a_val * B[kk * TK + k, jj * TJ + j]

# Apply scheduling primitives: parallelize the outermost tile loop over ii.
# Different ii-tiles update disjoint sets of rows in C, so this is safe.
kernel_gemm_opt = kernel_gemm_tiled
kernel_gemm_opt = parallelize_loop(kernel_gemm_opt, kernel_gemm_opt.find_loop("ii"))

# Optionally simplify expressions and control flow in the transformed kernel.
kernel_gemm_opt = simplify(kernel_gemm_opt)

# Rename to the public entry point expected by the C driver and build system.
kernel_gemm = rename(kernel_gemm_opt, "kernel_gemm")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_gemm (/*ctxt=*/NULL, ni, nj, nk,
	       (DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
	       (DATA_TYPE*)POLYBENCH_ARRAY(C),
	       (DATA_TYPE*)POLYBENCH_ARRAY(A),
	       (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}