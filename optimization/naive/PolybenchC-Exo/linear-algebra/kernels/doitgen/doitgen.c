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
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Baseline kernel: direct transcription of the PolyBench doitgen kernel.
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
            # Original PolyBench structure:
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


# Scheduling: improve data locality by separating initialization and computation
# and then interchanging the p and s loops so that we stream through C4 rows and
# the temporary sum[] buffer.
kernel_doitgen_opt = kernel_doitgen_base

# 1) Fission the p-loop inside the (r, q) nest into
#    (a) sum[p] initialization and (b) the reduction over s.
p_loop = kernel_doitgen_opt.find_loop("p")    # the first 'p' loop under (r, q)
p_body = p_loop.body()
gap = p_body[0].after()                       # between 'sum[p] = 0.0' and the 'for s' loop
kernel_doitgen_opt = fission(kernel_doitgen_opt, gap)

# After fission we have, for each (r, q):
#   for p in 0..np:     sum[p] = 0
#   for p in 0..np:     for s in 0..np: sum[p] += A[r,q,s] * C4[s,p]
#   for p in 0..np:     A[r,q,p] = sum[p]

# 2) Interchange the second p-loop with the inner s-loop so that we iterate
#    over s first and then p. This enables streaming access over C4 rows and
#    sum[], and reuses A[r,q,s] across all p.
kernel_doitgen_opt = reorder_loops(kernel_doitgen_opt, "p s")

# After reordering, for each (r, q) we effectively have:
#   for p in 0..np:     sum[p] = 0
#   for s in 0..np:
#       for p in 0..np:
#           sum[p] += A[r,q,s] * C4[s,p]
#   for p in 0..np:     A[r,q,p] = sum[p]
#
# This is algebraically and bitwise equivalent to the original loop nest for
# each element A[r,q,p], but has much better spatial locality on C4 and sum.

# 3) Cleanup any trivial induction variables or expressions.
kernel_doitgen_opt = simplify(kernel_doitgen_opt)

# 4) Expose the scheduled kernel under the name used by the C driver.
kernel_doitgen = rename(kernel_doitgen_opt, "kernel_doitgen")
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