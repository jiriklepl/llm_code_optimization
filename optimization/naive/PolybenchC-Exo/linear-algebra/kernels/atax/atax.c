/**
 * Exo ATAX driver: mirrors PolyBench/C atax.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"

/* Include the Exo-generated kernel header. */
#include "generated/atax/atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# Baseline ATAX kernel: y = A^T * (A * x)
@proc
def kernel_atax_unoptimized(
    m: size,
    n: size,
    A: DATA_TYPE[m, n] @ DRAM,
    x: DATA_TYPE[n] @ DRAM,
    y: DATA_TYPE[n] @ DRAM,
    tmp: DATA_TYPE[m] @ DRAM,
):
    # Initialize the output vector y once
    for j in seq(0, n):
        y[j] = 0.0

    # First pass: tmp = A * x (matrix–vector multiply over rows of A)
    for i in seq(0, m):
        tmp[i] = 0.0
        for j in seq(0, n):
            tmp[i] += A[i, j] * x[j]

    # Second pass: y = A^T * tmp
    # Each y[j] accumulates contributions from all rows i.
    for i in seq(0, m):
        for j in seq(0, n):
            y[j] += A[i, j] * tmp[i]


# ---------------------------------------------------------------------------
# Scheduling: create an optimized version of the kernel
# ---------------------------------------------------------------------------

kernel_atax_scheduled = kernel_atax_unoptimized

# 1) Parallelize the first matrix–vector multiply over rows of A (the second i-loop).
#    This loop computes tmp[i] independently for each i:
#      - writes: tmp[i]
#      - reads:  A[i, :], x[:]
#    There are no cross-iteration dependences, so it is safe to parallelize.
i_tmp = kernel_atax_scheduled.find_loop("i #1")
kernel_atax_scheduled = parallelize_loop(kernel_atax_scheduled, i_tmp)

# 2) Improve locality and vectorization on the first j-loop (inside tmp computation).
#    We split j into:
#      jo : outer tile index
#      ji : inner index within a tile of size 32
#    This keeps the innermost loop small with a constant trip count,
#    which is friendly to unrolling and SIMD vectorization.
j_tmp = kernel_atax_scheduled.find_loop("j #0")
kernel_atax_scheduled = divide_loop(
    kernel_atax_scheduled,
    j_tmp,
    32,                 # tile size
    ("jo", "ji"),       # new iterators
    tail="guard",       # handle n % 32 != 0 with guarded iterations
    perfect=False,
)

#    The inner loop over ji now has a constant upper bound (32), so we
#    fully unroll it to expose instruction-level parallelism and make
#    the access pattern to A[i, j] and x[j] as regular as possible.
ji_loop = kernel_atax_scheduled.find_loop("ji")
kernel_atax_scheduled = unroll_loop(kernel_atax_scheduled, ji_loop)

# 3) Parallelize the j-loop of the second pass.
#    For a fixed i, iterations over j touch disjoint elements of y:
#      - writes: y[j] for different j
#      - reads:  A[i, j], tmp[i]
#    Hence the j-loop has no cross-iteration dependences and can be
#    safely parallelized.
j_y = kernel_atax_scheduled.find_loop("j")
kernel_atax_scheduled = parallelize_loop(kernel_atax_scheduled, j_y)

# 4) Expose the scheduled kernel under the name expected by the C driver.
kernel_atax = rename(kernel_atax_scheduled, "kernel_atax")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten views to raw pointers. */
  kernel_atax (/*ctxt=*/NULL, m, n,
               (DATA_TYPE*)POLYBENCH_ARRAY(A),
               (DATA_TYPE*)POLYBENCH_ARRAY(x),
               (DATA_TYPE*)POLYBENCH_ARRAY(y),
               (DATA_TYPE*)POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}