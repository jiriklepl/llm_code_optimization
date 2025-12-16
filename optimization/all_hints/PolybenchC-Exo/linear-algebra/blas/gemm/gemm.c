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

/* Kernel implementation in Exo (optimized with tiling and parallelism):

EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *        # scheduling primitives
from exo.libs.memories import DRAM      # main memory annotation

# Base GEMM kernel:
# 1. Scale C by beta.
# 2. Accumulate alpha * A * B into C.
#
# DATA_TYPE is a placeholder that will be specialized to the concrete
# scalar type (e.g., float or double) when generating C.

@proc
def kernel_gemm(
    ni: size,
    nj: size,
    nk: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[ni, nj] @ DRAM,
    A: DATA_TYPE[ni, nk] @ DRAM,
    B: DATA_TYPE[nk, nj] @ DRAM,
):

    # Phase 1: scale C by beta. This keeps the scaling separate from the
    # matrix multiplication, making it easy to schedule the compute-heavy
    # part without duplicating the scaling.
    for i in seq(0, ni):
        for j in seq(0, nj):
            C[i, j] = beta * C[i, j]

    # Phase 2: C += alpha * A * B
    # Naive triple-nested loops over i, j, k.
    for i in seq(0, ni):
        for j in seq(0, nj):
            for k in seq(0, nk):
                C[i, j] += alpha * A[i, k] * B[k, j]


# ------------------------------
# Optimization schedule
# ------------------------------
#
# We optimize only the compute-heavy GEMM phase (the second i-loop nest).
# Transformations applied:
#   - 3D tiling over i, j, k to improve cache locality.
#   - Loop reordering to iterate over tiles in (io, jo, ko) order.
#     This keeps working tiles of C, A, and B hot in cache.
#   - Parallelization of the outermost i-tile loop with OpenMP.
#   - Simplification of the resulting IR.
#
# Tile sizes (can be tuned for the target machine’s cache hierarchy):

II = 64   # tile size in the i (row) dimension
JJ = 64   # tile size in the j (column) dimension
KK = 32   # tile size in the k (reduction) dimension

kernel_gemm_opt = kernel_gemm

# Tile the i dimension of the GEMM phase (second 'i' loop: "i #1").
i_gemm = kernel_gemm_opt.find_loop("i #1")
kernel_gemm_opt = divide_loop(kernel_gemm_opt, i_gemm, II, ["io", "ii"], tail="guard")
# After this:
#   for io in 0..ceil(ni/II):
#     for ii in 0..II:
#       if io*II + ii < ni:
#         for j in ...
#           for k in ...

# Tile the j dimension of the GEMM phase (second 'j' loop: "j #1").
j_gemm = kernel_gemm_opt.find_loop("j #1")
kernel_gemm_opt = divide_loop(kernel_gemm_opt, j_gemm, JJ, ["jo", "ji"], tail="guard")
# Structure inside GEMM now:
#   for io:
#     for ii:
#       if in-bounds(i):
#         for jo:
#           for ji:
#             if in-bounds(j):
#               for k:
#                 ...

# Tile the k (reduction) dimension.
k_gemm = kernel_gemm_opt.find_loop("k")
kernel_gemm_opt = divide_loop(kernel_gemm_opt, k_gemm, KK, ["ko", "ki"], tail="guard")
# Now GEMM loops look like:
#   for io:
#     for ii:
#       if in-bounds(i):
#         for jo:
#           for ji:
#             if in-bounds(j):
#               for ko:
#                 for ki:
#                   if in-bounds(k):
#                     C[i,j] += alpha * A[i,k] * B[k,j]

# Reorder loops so that we iterate tiles in (io, jo, ko) order, and then
# iterate within each tile over (ii, ji, ki).
# Step 1: bring jo outside ii: (io, ii, jo, ...) -> (io, jo, ii, ...)
kernel_gemm_opt = reorder_loops(kernel_gemm_opt, "ii jo")

# Step 2: interchange ji and ko so that k-tiles come before j-tiles inside
# each (io, jo, ii) region: (.., ji, ko, ki) -> (.., ko, ji, ki)
kernel_gemm_opt = reorder_loops(kernel_gemm_opt, "ji ko")

# Step 3: move ko outside ii so that final order is (io, jo, ko, ii, ji, ki):
# (io, jo, ii, ko, ...) -> (io, jo, ko, ii, ...)
kernel_gemm_opt = reorder_loops(kernel_gemm_opt, "ii ko")

# Parallelize the outermost i-tile loop. Each io-iteration updates a disjoint
# set of rows of C, so this is safe and exposes coarse-grain parallelism.
io_loop = kernel_gemm_opt.find_loop("io")
kernel_gemm_opt = parallelize_loop(kernel_gemm_opt, io_loop)

# Clean up any redundant control and simplify expressions generated by tiling.
kernel_gemm_opt = simplify(kernel_gemm_opt)

# Rename the optimized procedure back to "kernel_gemm" so that the generated
# C symbol matches the name used in this driver.
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