/**
 * Exo ADI driver: mirrors PolyBench/C adi.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"

/* Include the Exo-generated kernel header. */
#include "generated/adi/adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static 
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n)) 
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo (optimized):
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM


@proc
def kernel_adi(
    tsteps: size,
    n: size,
    mul1: DATA_TYPE,
    mul2: DATA_TYPE,
    u: DATA_TYPE[n, n] @ DRAM,
    v: DATA_TYPE[n, n] @ DRAM,
    p: DATA_TYPE[n, n] @ DRAM,
    q: DATA_TYPE[n, n] @ DRAM,
):
    """
    Optimized ADI kernel.

    Differences vs. the baseline Exo kernel:

    - The observable result is the updated field u; v is an intermediate,
      while p and q are pure scratch in the original PolyBench kernel.
      Nothing outside the kernel reads p or q.

    - Instead of storing the Thomas-coefficient arrays p[i,j], q[i,j] as
      full 2D grids, we use 1D temporaries per active line:

          p_col[j], q_col[j]  for column (x-direction) sweep
          p_row[j], q_row[j]  for row    (y-direction) sweep

      This keeps the working set for the tridiagonal solver to O(n) and
      avoids touching the large 2D scratch arrays p and q altogether.
      The driver still allocates p and q, but the kernel no longer
      reads or writes them, which reduces memory traffic.

    - We hoist and reuse denominators inside the Thomas forward sweeps,
      so each j-step performs a single division instead of two.

    The mathematical update of u (and v) is unchanged: for every
    time step, this kernel performs exactly the same ADI scheme as
    the original PolyBench implementation.
    """

    assert n > 2

    # ADI coefficients (loop-invariant)
    a: DATA_TYPE
    b: DATA_TYPE
    c: DATA_TYPE
    d: DATA_TYPE
    e: DATA_TYPE
    f: DATA_TYPE

    # Frequently reused combinations
    one_plus_2d: DATA_TYPE
    one_plus_2a: DATA_TYPE

    # Per-sweep denominators (reused within each j-iteration)
    denom_col: DATA_TYPE
    inv_denom_col: DATA_TYPE
    denom_row: DATA_TYPE
    inv_denom_row: DATA_TYPE

    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    one_plus_2d = 1.0 + 2.0 * d
    one_plus_2a = 1.0 + 2.0 * a

    # 1D temporaries for the Thomas algorithm along a single line in j.
    # They replace the logical role of p[i, j] and q[i, j] for each line.
    p_col: DATA_TYPE[n] @ DRAM
    q_col: DATA_TYPE[n] @ DRAM
    p_row: DATA_TYPE[n] @ DRAM
    q_row: DATA_TYPE[n] @ DRAM

    # Time-stepping loop: identical number of steps as in the original
    # (original used seq(1, tsteps + 1), which also executes tsteps times).
    for t in seq(0, tsteps):
        # --------------------------------------------------------------
        # Column sweep: for each interior column i, solve a tridiagonal
        # system along j using u(:, i-1..i+1) to produce v(:, i).
        #
        # This is exactly the same Thomas algorithm as in the baseline,
        # but the coefficients for a given column are stored in p_col/q_col.
        # --------------------------------------------------------------
        for i in seq(1, n - 1):
            # Dirichlet boundary at j = 0 for this column.
            v[0, i] = 1.0
            p_col[0] = 0.0
            q_col[0] = v[0, i]

            # Forward sweep along j: build modified coefficients for this column.
            for j in seq(1, n - 1):
                denom_col = a * p_col[j - 1] + b
                inv_denom_col = 1.0 / denom_col

                # p[i, j] = -c / (a * p[i, j-1] + b)
                p_col[j] = -c * inv_denom_col

                # q[i, j] =
                #   ( -d * u[j, i-1]
                #     + (1 + 2*d) * u[j, i]
                #     - f * u[j, i+1]
                #     - a * q[i, j-1]
                #   ) / (a * p[i, j-1] + b)
                q_col[j] = (
                    -d * u[j, i - 1]
                    + one_plus_2d * u[j, i]
                    - f * u[j, i + 1]
                    - a * q_col[j - 1]
                ) * inv_denom_col

            # Dirichlet boundary at j = n-1.
            v[n - 1, i] = 1.0

            # Backward substitution: recover interior v[:, i].
            # Original code: v[n-1-j, i] = p[i, n-1-j] * v[n-j, i] + q[i, n-1-j]
            for j in seq(1, n - 1):
                v[n - 1 - j, i] = (
                    p_col[n - 1 - j] * v[n - j, i]
                    + q_col[n - 1 - j]
                )

        # --------------------------------------------------------------
        # Row sweep: for each interior row i, solve a tridiagonal system
        # along j using v(i-1..i+1, :) to update u(i, :).
        #
        # Again, we perform the same Thomas algorithm as the baseline,
        # but store coefficients in p_row/q_row instead of p[i, j]/q[i, j].
        # --------------------------------------------------------------
        for i in seq(1, n - 1):
            # Dirichlet boundary at j = 0 for this row.
            u[i, 0] = 1.0
            p_row[0] = 0.0
            q_row[0] = u[i, 0]

            # Forward sweep along j for this row.
            for j in seq(1, n - 1):
                denom_row = d * p_row[j - 1] + e
                inv_denom_row = 1.0 / denom_row

                # p[i, j] = -f / (d * p[i, j-1] + e)
                p_row[j] = -f * inv_denom_row

                # q[i, j] =
                #   ( -a * v[i-1, j]
                #     + (1 + 2*a) * v[i, j]
                #     - c * v[i+1, j]
                #     - d * q[i, j-1]
                #   ) / (d * p[i, j-1] + e)
                q_row[j] = (
                    -a * v[i - 1, j]
                    + one_plus_2a * v[i, j]
                    - c * v[i + 1, j]
                    - d * q_row[j - 1]
                ) * inv_denom_row

            # Dirichlet boundary at j = n-1.
            u[i, n - 1] = 1.0

            # Backward substitution along j: update interior of u[i, :].
            # Original code: u[i, n-1-j] = p[i, n-1-j] * u[i, n-j] + q[i, n-1-j]
            for j in seq(1, n - 1):
                u[i, n - 1 - j] = (
                    p_row[n - 1 - j] * u[i, n - j]
                    + q_row[n - 1 - j]
                )
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Precompute scalar coefficients used by the Exo kernel. */
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;

  DX = SCALAR_VAL(1.0) / (DATA_TYPE)n;
  DY = SCALAR_VAL(1.0) / (DATA_TYPE)n;
  DT = SCALAR_VAL(1.0) / (DATA_TYPE)tsteps;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_adi(/*ctxt=*/NULL,
             tsteps, n,
             (DATA_TYPE*)&mul1, (DATA_TYPE*)&mul2,
             (DATA_TYPE*)POLYBENCH_ARRAY(u),
             (DATA_TYPE*)POLYBENCH_ARRAY(v),
             (DATA_TYPE*)POLYBENCH_ARRAY(p),
             (DATA_TYPE*)POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}