/**
 * Exo GEMVER driver: mirrors PolyBench/C gemver.c but calls the Exo-generated kernel.
 *
 * Original authors:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

/* Include the Exo-generated kernel header. */
#include "generated/gemver/gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *
from exo.API_cursors import ForCursor


# Baseline GEMVER kernel, matching the PolyBench reference semantics.
@proc
def gemver_unoptimized(
    n: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    A: DATA_TYPE[n, n] @ DRAM,
    u1: DATA_TYPE[n] @ DRAM,
    v1: DATA_TYPE[n] @ DRAM,
    u2: DATA_TYPE[n] @ DRAM,
    v2: DATA_TYPE[n] @ DRAM,
    w: DATA_TYPE[n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
    z: DATA_TYPE[n] @ DRAM,
):
    # 1) Rank-2 update: A = A + u1*v1^T + u2*v2^T
    for i in seq(0, n):
        for j in seq(0, n):
            A[i, j] += u1[i] * v1[j] + u2[i] * v2[j]

    # 2) x = x + beta * A^T * y
    for i in seq(0, n):
        for j in seq(0, n):
            x[i] += beta * A[j, i] * y[j]

    # 3) x = x + z
    for i in seq(0, n):
        x[i] += z[i]

    # 4) w = w + alpha * A * x
    for i in seq(0, n):
        for j in seq(0, n):
            w[i] += alpha * A[i, j] * x[j]


# ---------------------------------------------------------------------------
# Scheduling: expose parallelism and improve memory locality.
#
# - Parallelize the outer i-loop of the rank-2 update (step 1):
#   each iteration updates an independent row of A.
#
# - For the x update (step 2), which computes
#       x[i] += beta * sum_j A[j, i] * y[j]
#   we:
#     * tile the i-dimension,
#     * reorder loops to (i-tile, j, i-in-tile) so A[j, i] is accessed
#       with i as the innermost index (row-major friendly),
#     * parallelize over i-tiles, which touch disjoint segments of x[].
#
# - Parallelize the outer i-loop of the w update (step 4), as each
#   iteration updates a distinct entry of w[].
# ---------------------------------------------------------------------------

# Tile size along i when forming x; chosen to balance cache reuse
# of a row of A against parallel granularity.
T_I = 32

p = gemver_unoptimized

# 1) Parallelize the outer i-loop of the rank-2 update on A.
i0 = p.find_loop("i")      # first 'for i in _:_'
p = parallelize_loop(p, i0)

# 2) Optimize the x update: tile i, reorder for better A locality,
#    and parallelize the outer tile loop.

# Locate the outer 'for i' that governs the first x[i] reduction.
x_red = p.find("x[i] += _")
cur = x_red
while not (isinstance(cur, ForCursor) and cur.name() == "i"):
    cur = cur.parent()
x_i_loop = cur

# Tile the i-loop into (io, ii) with a cut tail so that we avoid
# data-dependent guards inside the loop body.
p = divide_loop(p, x_i_loop, T_I, ("io", "ii"), tail="cut")

# After tiling there are two (ii, j) nests (main block and tail block).
# Reorder each from (ii, j) to (j, ii) so that the innermost loop walks
# A[j, i] over contiguous i, improving cache and vectorization.
p = reorder_loops(p, "ii j")
p = reorder_loops(p, "ii j #1")

# Parallelize the outer tile loop over io.  Each tile works on a
# disjoint slice of x[], so there are no cross-tile dependences.
io_loop = p.find_loop("io")
p = parallelize_loop(p, io_loop)

# 3) Parallelize the outer i-loop of the final w update.
w_red = p.find("w[i] += _")
cur = w_red
while not (isinstance(cur, ForCursor) and cur.name() == "i"):
    cur = cur.parent()
w_i_loop = cur
p = parallelize_loop(p, w_i_loop)

# 4) Expose the scheduled kernel under the name expected by the C driver.
kernel_gemver = rename(p, "kernel_gemver")

EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y,  DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z,  DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to raw pointers. */
  kernel_gemver(/*ctxt=*/NULL, n,
                (DATA_TYPE*)&alpha,
                (DATA_TYPE*)&beta,
                (DATA_TYPE*)POLYBENCH_ARRAY(A),
                (DATA_TYPE*)POLYBENCH_ARRAY(u1),
                (DATA_TYPE*)POLYBENCH_ARRAY(v1),
                (DATA_TYPE*)POLYBENCH_ARRAY(u2),
                (DATA_TYPE*)POLYBENCH_ARRAY(v2),
                (DATA_TYPE*)POLYBENCH_ARRAY(w),
                (DATA_TYPE*)POLYBENCH_ARRAY(x),
                (DATA_TYPE*)POLYBENCH_ARRAY(y),
                (DATA_TYPE*)POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}