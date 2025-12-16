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

# Base, straightforward GEMM specification.
#   C[ni,nj] = beta * C + alpha * A[ni,nk] * B[nk,nj]
#
# We write a simple, readable version and then apply scheduling
# transformations below to obtain an optimized 'kernel_gemm'.

@proc
def kernel_gemm_base(
    ni: size,
    nj: size,
    nk: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[ni, nj] @ DRAM,
    A: DATA_TYPE[ni, nk] @ DRAM,
    B: DATA_TYPE[nk, nj] @ DRAM,
):
    # For each output element C[i, j], scale by beta and then accumulate
    # alpha * A[i, k] * B[k, j] over k.
    for i in seq(0, ni):
        for j in seq(0, nj):
            C[i, j] = beta * C[i, j]
            for k in seq(0, nk):
                C[i, j] += alpha * A[i, k] * B[k, j]


# -------------------------------------------------------------------
# Scheduling: tiling + parallelization
# -------------------------------------------------------------------
# We transform 'kernel_gemm_base' into an optimized kernel:
#  - Tile i and j to work on ni-by-nj tiles of C, which improves
#    cache locality for A, B, and C.
#  - Optionally tile k to reduce working-set size along the reduction
#    dimension.
#  - Parallelize the outermost tiled-i loop (io) with OpenMP.
#
# These transformations preserve the mathematical computation but
# change loop structure to be more cache- and core-friendly.

kernel_gemm = kernel_gemm_base

# Tunable tile sizes – can be adjusted for a given machine.
Ti = 64  # tile size in the i (row) dimension
Tj = 64  # tile size in the j (column) dimension
Tk = 32  # tile size along the reduction dimension k

# 1) Tile the i loop: i -> (io, ii)
#    This creates blocks of Ti consecutive rows of C and A.
kernel_gemm = divide_loop(kernel_gemm, "i", Ti, ("io", "ii"), tail="guard")

# 2) Tile the j loop: j -> (jo, jj)
#    This creates blocks of Tj consecutive columns of C and B.
kernel_gemm = divide_loop(kernel_gemm, "j", Tj, ("jo", "jj"), tail="guard")

# After steps (1) and (2), the loop order is roughly:
#   for io:
#     for ii:
#       for jo:
#         for jj:
#           ...
# We want to iterate tiles of C in (io, jo) order, and then iterate
# within each tile with (ii, jj), so we swap the middle two loops.
kernel_gemm = reorder_loops(kernel_gemm, "ii jo")

# 3) Tile the reduction loop k: k -> (ko, ki)
#    This limits the size of the working set touched across k at once,
#    which can help cache behavior when nk is large.
kernel_gemm = divide_loop(kernel_gemm, "k", Tk, ("ko", "ki"), tail="guard")

# 4) Parallelize across tiles of rows (io).
#    Iterations over different io are independent: they operate on
#    disjoint row ranges of C and A and only read B.
#    Exo will lower this to an OpenMP parallel for loop.
kernel_gemm = parallelize_loop(kernel_gemm, "io")

# 5) Clean up expressions and eliminate any trivial conditionals
#    introduced by guarded tiling.
kernel_gemm = simplify(kernel_gemm)

# 6) Rename to the external name expected by the C driver.
kernel_gemm = rename(kernel_gemm, "kernel_gemm")
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