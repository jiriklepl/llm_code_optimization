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

/* Optimized kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM


# Baseline 2mm kernel expressed in Exo.
# We then apply a schedule that:
#   - rearranges loops for better memory access (inner loops walk
#     contiguous j-dimension where possible),
#   - tiles the inner j loops to improve cache locality, and
#   - parallelizes the outer i loops with OpenMP.
@proc
def kernel_2mm_base(
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
    # D := alpha * A * B * C + beta * D

    # Basic sanity: all problem sizes must be positive.
    assert ni >= 1
    assert nj >= 1
    assert nk >= 1
    assert nl >= 1

    # 1) Initialize tmp = 0.
    #    Walk tmp row-major (i, then j) to write contiguous elements.
    for i in seq(0, ni):
        for j in seq(0, nj):
            tmp[i, j] = 0.0

    # 2) tmp := alpha * A * B
    #
    # Loop order i-k-j:
    #   - k is the reduction dimension
    #   - j is innermost so that B[k, j] and tmp[i, j] are accessed
    #     with unit stride along their last dimension.
    for i in seq(0, ni):
        for k in seq(0, nk):
            for j in seq(0, nj):
                tmp[i, j] += alpha * A[i, k] * B[k, j]

    # 3) Scale D by beta.
    #    Again use (i, j) so that D[i, j] is accessed contiguously in j.
    for i in seq(0, ni):
        for j in seq(0, nl):
            D[i, j] = beta * D[i, j]

    # 4) D += tmp * C
    #
    # Loop order i-k-j:
    #   - k is the reduction dimension
    #   - j is innermost so that C[k, j] and D[i, j] are unit-stride.
    for i in seq(0, ni):
        for k in seq(0, nj):
            for j in seq(0, nl):
                D[i, j] += tmp[i, k] * C[k, j]


# ------------------------ Scheduling (optimization) -------------------------

# Tunable tile size for the j/nj and j/nl dimensions.
# This can be adjusted to match the cache hierarchy of the target machine.
J_BLOCK = 64

kernel_2mm_sched = kernel_2mm_base

# Tile the j-loop of the tmp accumulation (the second j-loop overall).
# This limits the working set of tmp and B per k-iteration to a j-tile.
kernel_2mm_sched = divide_loop(
    kernel_2mm_sched,
    "j #1",                # j in: for i in ni: for k in nk: for j in nj
    J_BLOCK,
    ["j_outer_tmp", "j_inner_tmp"],
    tail="guard",
)

# Tile the j-loop of the D accumulation (the fourth j-loop overall).
# This limits the working set of D and C per k-iteration to a j-tile.
kernel_2mm_sched = divide_loop(
    kernel_2mm_sched,
    "j #3",                # j in: for i in ni: for k in nj: for j in nl
    J_BLOCK,
    ["j_outer_D", "j_inner_D"],
    tail="guard",
)

# Parallelize across i for all four outer loops. Each i-iteration operates
# on a distinct row of tmp and D, so there are no cross-iteration dependences.
kernel_2mm_sched = parallelize_loop(kernel_2mm_sched, "i #0")  # tmp init
kernel_2mm_sched = parallelize_loop(kernel_2mm_sched, "i #1")  # tmp update
kernel_2mm_sched = parallelize_loop(kernel_2mm_sched, "i #2")  # D scale
kernel_2mm_sched = parallelize_loop(kernel_2mm_sched, "i #3")  # D update

# Simplify expressions and clean up any dead code introduced by tiling.
kernel_2mm_sched = simplify(kernel_2mm_sched)

# Export the final scheduled kernel under the name expected by the C driver.
kernel_2mm = rename(kernel_2mm_sched, "kernel_2mm")
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