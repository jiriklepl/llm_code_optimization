/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"

/* Include the Exo-generated kernel header. */
#include "generated/symm/symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
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

# Column-tile size along j. This should be tuned for the target machine,
# but a moderate default (e.g., 64) gives good cache behavior while keeping
# the per-tile scratch buffer small.
TILE_J = 64

@proc
def kernel_symm(
    m: size,
    n: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[m, n] @ DRAM,
    A: DATA_TYPE[m, m] @ DRAM,
    B: DATA_TYPE[m, n] @ DRAM,
):
    """
    Optimized symmetric matrix-matrix multiply kernel.

    Matrices:
      - A: m x m, row-major, only lower triangle (k <= i) is initialized/used.
      - B: m x n, row-major.
      - C: m x n, row-major.

    Semantics (matches PolyBench SYMM):
        C := beta * C + alpha * A_sym * B

    where A_sym is the symmetric matrix induced by the lower triangle of A.
    This implementation:
      - touches only A[i, k] for k <= i (never reads the -999 upper triangle),
      - uses a symmetry-exploiting algorithm to avoid referencing A[k, i],
      - traverses B and C with unit-stride along j for good locality,
      - tiles columns in chunks of TILE_J to improve cache reuse,
      - uses a small per-tile scratch buffer to accumulate "temp2" values.
    """

    # -------------------------------------------------------------------------
    # 1. Pre-scale C by beta: C[i, j] = beta * C[i, j]
    #
    # This is algebraically equivalent to the original kernel, which multiplies
    # each C[i, j] by beta exactly once (at the i-th outer iteration) before
    # adding all alpha*A*B contributions. Doing this as a separate pass
    # exposes a pure accumulation form in the main kernel.
    # We tile over j for better cache behavior.
    # -------------------------------------------------------------------------
    for J in seq(0, (n + TILE_J - 1) // TILE_J):
        for i in seq(0, m):
            for jj in seq(0, TILE_J):
                j = J * TILE_J + jj
                if j < n:
                    C[i, j] = beta * C[i, j]

    # -------------------------------------------------------------------------
    # 2. Symmetric matrix-matrix multiply:
    #
    #    For each row i:
    #      - For all k < i, we:
    #          * update row k of C using A[i, k] and row i of B,
    #          * accumulate into a per-row temporary vector temp2(j) the
    #            contributions from A[i, k] * B[k, j].
    #      - Finally, we update row i of C using A[i, i] and temp2(j).
    #
    #    In untiled, scalar form this is:
    #
    #      for i in 0..m-1:
    #          for j in 0..n-1:
    #              temp2[j] = 0
    #          for k in 0..i-1:
    #              a_ik = A[i, k]
    #              for j in 0..n-1:
    #                  C[k, j] += alpha * B[i, j] * a_ik
    #                  temp2[j] += B[k, j] * a_ik
    #          for j in 0..n-1:
    #              C[i, j] += alpha * (B[i, j] * A[i, i] + temp2[j])
    #
    #    This is exactly equivalent to:
    #      C[i, j] = beta * C0[i, j] + alpha * sum_k A_sym[i, k] * B[k, j]
    #
    #    We tile over j to keep a small working set in cache. For each
    #    column tile J, we use a TILE_J-element scratch buffer temp2,
    #    so extra storage is O(TILE_J) per (i, J), far less than 50% of
    #    the base footprint.
    # -------------------------------------------------------------------------
    for J in seq(0, (n + TILE_J - 1) // TILE_J):
        for i in seq(0, m):
            # Per-(i, J) temporary accumulator over columns in this tile:
            #   temp2[jj] ≈ temp2(global_j = J*TILE_J + jj)
            temp2: DATA_TYPE[TILE_J] @ DRAM

            # Initialize temp2 for all columns in this tile that are in-bounds.
            for jj in seq(0, TILE_J):
                j = J * TILE_J + jj
                if j < n:
                    temp2[jj] = 0

            # Accumulate strictly lower-triangular contributions: k < i.
            # We use only the stored lower triangle A[i, k] with k < i.
            for k in seq(0, i):
                a_ik: DATA_TYPE
                a_ik = A[i, k]

                # For this (i, k) pair, walk all columns in the current tile.
                # Accesses:
                #   - B[i, j] and B[k, j] are row-major, contiguous in j.
                #   - C[k, j] is row-major, contiguous in j.
                #   - temp2[jj] is contiguous in jj.
                for jj in seq(0, TILE_J):
                    j = J * TILE_J + jj
                    if j < n:
                        # Contribution to row k:
                        #   C[k, j] += alpha * B[i, j] * A[i, k]
                        C[k, j] += alpha * B[i, j] * a_ik

                        # Accumulate contribution to row i via temp2(j):
                        #   temp2(j) += B[k, j] * A[i, k]
                        temp2[jj] += B[k, j] * a_ik

            # Finish row i for columns in this tile using the diagonal A[i, i]
            # and the accumulated temp2.
            for jj in seq(0, TILE_J):
                j = J * TILE_J + jj
                if j < n:
                    C[i, j] += alpha * (B[i, j] * A[i, i] + temp2[jj])

# Apply a simple simplification pass to clean up expressions before codegen.
kernel_symm = simplify(kernel_symm)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_symm (/*ctxt=*/NULL, m, n,
	       (DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
	       (DATA_TYPE*)POLYBENCH_ARRAY(C),
	       (DATA_TYPE*)POLYBENCH_ARRAY(A),
	       (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}