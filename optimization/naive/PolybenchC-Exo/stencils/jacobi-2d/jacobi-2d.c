/**
 * Exo Jacobi-2D driver: mirrors PolyBench/C jacobi-2d.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Include the Exo-generated kernel header. */
#include "generated/jacobi-2d/jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
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


# Baseline Jacobi-2D kernel: matches the PolyBench reference algorithm.
@proc
def kernel_jacobi_2d(
    tsteps: size,
    n: size,
    A: DATA_TYPE[n, n] @ DRAM,
    B: DATA_TYPE[n, n] @ DRAM,
):
    # We update only interior points [1 : n-1), so n must be > 2.
    assert n > 2

    for t in seq(0, tsteps):
        # Update B from A.
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                B[i, j] = 0.2 * (
                    A[i, j]
                    + A[i, j - 1]
                    + A[i, j + 1]
                    + A[i + 1, j]
                    + A[i - 1, j]
                )

        # Update A from B.
        for i in seq(1, n - 1):
            for j in seq(1, n - 1):
                A[i, j] = 0.2 * (
                    B[i, j]
                    + B[i, j - 1]
                    + B[i, j + 1]
                    + B[i + 1, j]
                    + B[i - 1, j]
                )


# Create an optimized version of the kernel using Exo's scheduling primitives.
# The optimization strategy is:
#   1. Mark both spatial i-loops as parallel (safe OpenMP parallel-for).
#   2. Strip-mine the inner j-loop in each sweep with a small constant factor
#      so that the innermost loop has a fixed trip count.
#   3. Fully unroll these small inner loops to expose instruction-level
#      parallelism and enable the backend to vectorize more aggressively.
BLOCK_J: int = 4

kernel_jacobi_2d_opt = kernel_jacobi_2d

# 1) Parallelize the two i-loops (one in each sweep).
#    There are exactly two loops with iterator 'i' at this point.
kernel_jacobi_2d_opt = parallelize_loop(kernel_jacobi_2d_opt, "i")
kernel_jacobi_2d_opt = parallelize_loop(kernel_jacobi_2d_opt, "i #1")

# 2) Strip-mine the j-loop of the first sweep (A -> B) into
#    j_outer_B (tile index) and j_inner_B (within-tile index).
kernel_jacobi_2d_opt = divide_loop(
    kernel_jacobi_2d_opt,
    "j",
    BLOCK_J,
    ("j_outer_B", "j_inner_B"),
    tail="guard",
)

# 3) Strip-mine the j-loop of the second sweep (B -> A) similarly.
kernel_jacobi_2d_opt = divide_loop(
    kernel_jacobi_2d_opt,
    "j",
    BLOCK_J,
    ("j_outer_A", "j_inner_A"),
    tail="guard",
)

# 4) Fully unroll the fixed-size inner j loops in both sweeps.
for c in kernel_jacobi_2d_opt.find_all("for j_inner_B in _:_"):
    kernel_jacobi_2d_opt = unroll_loop(kernel_jacobi_2d_opt, c)

for c in kernel_jacobi_2d_opt.find_all("for j_inner_A in _:_"):
    kernel_jacobi_2d_opt = unroll_loop(kernel_jacobi_2d_opt, c)

# 5) Use the optimized version as the exported kernel.
kernel_jacobi_2d = kernel_jacobi_2d_opt
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_jacobi_2d(/*ctxt=*/NULL,
                   tsteps, n,
                   (DATA_TYPE*)POLYBENCH_ARRAY(A),
                   (DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}