/**
 * Exo DOITGEN driver: mirrors PolyBench/C doitgen.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"

/* Include the Exo-generated kernel header. */
#include "generated/doitgen/doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Baseline DOITGEN kernel that mirrors the original PolyBench implementation.
# This version has suboptimal locality for C4 because it walks C4 by columns.
@proc
def kernel_doitgen_base(
    nr: size,
    nq: size,
    np: size,
    A: DATA_TYPE[nr, nq, np] @ DRAM,
    C4: DATA_TYPE[np, np] @ DRAM,
    sum: DATA_TYPE[np] @ DRAM,
):
    for r in seq(0, nr):
        for q in seq(0, nq):
            # Original PolyBench-like structure:
            #   for p:
            #     sum[p] = 0
            #     for s:
            #       sum[p] += A[r,q,s] * C4[s,p]
            #   for p:
            #     A[r,q,p] = sum[p]
            for p in seq(0, np):
                sum[p] = 0.0
                for s in seq(0, np):
                    sum[p] += A[r, q, s] * C4[s, p]
            for p in seq(0, np):
                A[r, q, p] = sum[p]

# Start from the baseline kernel and apply locality- and SIMD-friendly
# scheduling transformations. The final procedure that will be compiled
# and called from C is `kernel_doitgen`.
kernel_doitgen = kernel_doitgen_base

# -------------------------------------------------------------------------
# 1) Separate zeroing of sum[p] from the accumulation.
#
# Inside each (r, q), the body currently is:
#   for p:
#       sum[p] = 0.0
#       for s:
#           sum[p] += ...
#
# We fission the inner p-loop around the gap between the assignment and
# the inner s-loop, so we get:
#   for p: sum[p] = 0.0
#   for p: for s: sum[p] += ...
#   for p: A[r,q,p] = sum[p]
#
# This preserves semantics but exposes the (p, s) nest we want to reorder.
p_init_loop = kernel_doitgen.find_loop("p")          # first 'p' loop
gap_after_init = p_init_loop.body()[0].after()       # after 'sum[p] = 0.0'
kernel_doitgen = fission(kernel_doitgen, gap_after_init, n_lifts=1)

# -------------------------------------------------------------------------
# 2) Improve C4 access locality by reordering (p, s) -> (s, p).
#
# After fission we have, for each (r, q):
#   for p: sum[p] = 0.0
#   for p:
#       for s:
#           sum[p] += A[r,q,s] * C4[s,p]
#   for p: A[r,q,p] = sum[p]
#
# In the accumulation loop the inner pattern is:
#   for p:
#       for s:
#           sum[p] += A[r,q,s] * C4[s,p]
#
# We reorder these two nested loops so that s becomes outer and p inner:
#   for s:
#       for p:
#           sum[p] += A[r,q,s] * C4[s,p]
#
# This changes nothing mathematically but:
#   - A[r,q,s] is now loaded once per s and reused across all p.
#   - C4[s,p] is traversed row-wise with unit-stride in p, matching
#     its row-major layout and encouraging SIMD.
accum_p_loop = kernel_doitgen.find_loop("p #1")      # second 'p' loop (the reduction)
kernel_doitgen = reorder_loops(kernel_doitgen, accum_p_loop)

# The resulting inner structure for each (r, q) is now:
#   for p in seq(0, np):
#       sum[p] = 0.0
#   for s in seq(0, np):
#       for p in seq(0, np):
#           sum[p] += A[r, q, s] * C4[s, p]
#   for p in seq(0, np):
#       A[r, q, p] = sum[p]
#
# This matches the optimized mathematical model:
#   sum[p] = Σ_s A[r,q,s] * C4[s,p]
# and preserves the final state of both A and sum while greatly reducing
# redundant loads from A and giving contiguous accesses to C4 and sum.
#
# We rely on the C compiler (with -O3) to auto-vectorize the innermost
# p-loop thanks to its simple, unit-stride access pattern, while Exo
# guarantees the correctness of the loop transformations above.

EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to 1D pointers. */
  kernel_doitgen (/*ctxt=*/NULL, nr, nq, np,
		  (DATA_TYPE*)POLYBENCH_ARRAY(A),
		  (DATA_TYPE*)POLYBENCH_ARRAY(C4),
		  (DATA_TYPE*)POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}