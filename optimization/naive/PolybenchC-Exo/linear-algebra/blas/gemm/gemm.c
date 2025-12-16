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

# Tile sizes for the i (rows of C/A) and j (columns of C/B) dimensions.
# These only appear in the scheduling code, not in the object code.
TI = 32
TJ = 32


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
    # Basic sanity; PolyBench always uses positive sizes.
    assert ni > 0
    assert nj > 0
    assert nk > 0

    # First scale C by beta.
    # This matches the first loop nest in the original PolyBench GEMM.
    for i in seq(0, ni):
        for j in seq(0, nj):
            C[i, j] = beta * C[i, j]

    # Main GEMM update:
    #   C[i,j] += alpha * sum_k A[i,k] * B[k,j]
    #
    # Loop order i,k,j gives good row-major access to B and C (j innermost).
    for i in seq(0, ni):
        for k in seq(0, nk):
            for j in seq(0, nj):
                C[i, j] += alpha * A[i, k] * B[k, j]


# -------------------------------------------------------------------------
# Scheduling: cache- and parallel-friendly blocked GEMM
# -------------------------------------------------------------------------

scheduled = kernel_gemm_base

# 1. Tile the i-dimension of the GEMM update (the second `i` loop).
#    We split it into:
#      io : outer tile index over rows
#      ii : row within a tile
i_gemm = scheduled.find_loop("i #1")  # second 'i' loop (the GEMM part)
scheduled = divide_loop(scheduled, i_gemm, TI, ("io", "ii"), tail="guard")

# 2. Tile the innermost j loop of the GEMM update.
#    We split it into:
#      jo : outer tile index over columns
#      jj : column within a tile
j_gemm = scheduled.find_loop("j #1")  # second 'j' loop (inside GEMM)
scheduled = divide_loop(scheduled, j_gemm, TJ, ("jo", "jj"), tail="guard")

# After the two divide_loop calls, the GEMM nest has the approximate
# structure:
#   for io
#     for ii
#       for k
#         for jo
#           for jj
#             C[...] += alpha * A[...] * B[...]

# 3. Reorder loops to operate on rectangular tiles of C and B:
#    Target order for the GEMM part:
#      io (tile of rows)
#      jo (tile of columns)
#      k  (reduction dimension)
#      ii (row within tile)
#      jj (col within tile)
#
#    This traversal:
#      * works on smaller C submatrices, improving cache locality,
#      * keeps j (via jj) as the innermost dimension for contiguous
#        row-major accesses to C and B,
#      * exposes good structure for autovectorization.
scheduled = reorder_loops(scheduled, "k jo")
scheduled = reorder_loops(scheduled, "ii jo")
scheduled = reorder_loops(scheduled, "ii k")

# 4. Parallelize the outermost tile loop over rows (io).
#    Each io-iteration touches disjoint rows of C, so this is safe and
#    gives multi-core parallelism via OpenMP.
io_outer = scheduled.find_loop("io")
scheduled = parallelize_loop(scheduled, io_outer)

# 5. Optionally simplify the resulting code (constant folding, dead
#    branches, etc.) and rename back to the name expected by the driver.
scheduled = simplify(scheduled)
kernel_gemm = rename(scheduled, "kernel_gemm")
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