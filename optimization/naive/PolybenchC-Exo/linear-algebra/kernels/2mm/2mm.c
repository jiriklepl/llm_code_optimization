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
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Tile size for the inner j dimension in the two matrix-multiply phases.
# This improves cache locality for accesses to B[k, j], C[k, j], D[i, j], tmp[i, j].
TILE_J = 32


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

    # First matrix multiply: tmp = alpha * A * B
    for i in seq(0, ni):
        for j in seq(0, nj):
            tmp[i, j] = 0.0
            for k in seq(0, nk):
                tmp[i, j] += alpha * A[i, k] * B[k, j]

    # Second matrix multiply and accumulation: D = beta * D + tmp * C
    for i in seq(0, ni):
        for j in seq(0, nl):
            D[i, j] = beta * D[i, j]
            for k in seq(0, nj):
                D[i, j] += tmp[i, k] * C[k, j]


# Scheduling to improve locality and parallelism while preserving semantics.

p = kernel_2mm_base

# 1. Stage 1 (tmp computation): separate initialization and accumulation.
#    Original:
#      for i:
#        for j:
#          tmp[i, j] = 0
#          for k: tmp[i, j] += ...
#    After fission (n_lifts=2) we get:
#      for i: for j: tmp[i, j] = 0
#      for i: for j: for k: tmp[i, j] += ...
c_tmp_init = p.find("tmp[_, _] = 0.0")
p = fission(p, c_tmp_init.after(), n_lifts=2)

# 2. Reorder loops in the tmp accumulation to i-k-j so that B[k, j]
#    is accessed with j innermost (contiguous in memory).
c_tmp_add = p.find("tmp[_, _] += _")
loop_k = c_tmp_add.parent()      # for k
loop_j = loop_k.parent()         # for j
p = reorder_loops(p, loop_j)     # swap j and k -> for i: for k: for j:

# 3. Stage 2 (D update): separate scaling of D and accumulation from tmp * C.
#    Original:
#      for i:
#        for j:
#          D[i, j] = beta * D[i, j]
#          for k: D[i, j] += ...
#    After fission (n_lifts=2):
#      for i: for j: D[i, j] = beta * D[i, j]
#      for i: for j: for k: D[i, j] += ...
c_D_scale = p.find("D[_, _] = beta * D[_, _]")
p = fission(p, c_D_scale.after(), n_lifts=2)

# 4. Reorder loops in the D accumulation to i-k-j so that C[k, j]
#    and D[i, j] are accessed with j innermost (contiguous).
c_D_add = p.find("D[_, _] += _")
loop_k2 = c_D_add.parent()       # for k
loop_j2 = loop_k2.parent()       # for j
p = reorder_loops(p, loop_j2)    # swap j and k -> for i: for k: for j:

# 5. Tile the j dimension in the two matrix-multiply-like loops.
#    This improves cache locality on the second (column) dimension.
#
#    5a. Tile j in the tmp accumulation loop.
c_tmp_add = p.find("tmp[_, _] += _")
loop_j1 = c_tmp_add.parent()       # inner j loop of stage-1 accumulation
p = divide_loop(p, loop_j1, TILE_J, ("jo1", "ji1"), tail="guard")

#    5b. Tile j in the D accumulation loop.
c_D_add = p.find("D[_, _] += _")
loop_j3 = c_D_add.parent()         # inner j loop of stage-2 accumulation
p = divide_loop(p, loop_j3, TILE_J, ("jo2", "ji2"), tail="guard")

# 6. Parallelize the outer i loops of the two compute-intensive phases.
#    These loops are independent across i (each i indexes distinct rows of
#    tmp, A, and D), so OpenMP-style parallelization is safe.
p = parallelize_loop(p, p.find_loop("i #1"))   # stage-1 accumulation over i
p = parallelize_loop(p, p.find_loop("i #3"))   # stage-2 accumulation over i

# 7. Simplify the resulting IR and give the final procedure the expected name.
p = simplify(p)
kernel_2mm = rename(p, "kernel_2mm")
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