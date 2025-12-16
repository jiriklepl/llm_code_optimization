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


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM

# ---------------------------------------------------------------------------
# Base ADI kernel, written to match PolyBench/C adi.c exactly.
# ---------------------------------------------------------------------------

@proc
def kernel_adi_base(
    tsteps: size,
    n: size,
    mul1: DATA_TYPE,
    mul2: DATA_TYPE,
    u: DATA_TYPE[n, n] @ DRAM,
    v: DATA_TYPE[n, n] @ DRAM,
    p: DATA_TYPE[n, n] @ DRAM,
    q: DATA_TYPE[n, n] @ DRAM,
):
    assert n > 2

    a: DATA_TYPE
    b: DATA_TYPE
    c: DATA_TYPE
    d: DATA_TYPE
    e: DATA_TYPE
    f: DATA_TYPE

    # Precompute coefficients as in PolyBench adi.c
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    # Time stepping loop
    for t in seq(0, tsteps):
        # ------------------------------------------------------------------
        # Column sweep: solve tridiagonal systems along the second dimension
        # for each interior row i. Memory accesses in the inner j-loop are
        # contiguous in the second index (good spatial locality).
        # ------------------------------------------------------------------
        for i in seq(1, n - 1):
            v[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = v[i, 0]

            # Forward sweep in j (j = 1 .. n-2)
            for j in seq(1, n - 1):
                p[i, j] = -a / (b + c * p[i, j - 1])
                q[i, j] = (
                    -d * u[i - 1, j]
                    + (1.0 + 2.0 * d) * u[i, j]
                    - f * u[i + 1, j]
                    - c * q[i, j - 1]
                ) / (b + c * p[i, j - 1])

            v[i, n - 1] = 1.0

            # Backward substitution in j (equivalent to j = n-2 .. 0)
            for j in seq(0, n - 1):
                v[i, n - 2 - j] = (
                    p[i, n - 2 - j] * v[i, n - 1 - j]
                    + q[i, n - 2 - j]
                )

        # ------------------------------------------------------------------
        # Row sweep: solve tridiagonal systems along the first dimension
        # for each interior column i. Again, the inner j-loop accesses v
        # contiguously in the second index.
        # ------------------------------------------------------------------
        for i in seq(1, n - 1):
            u[0, i] = 1.0
            p[0, i] = 0.0
            q[0, i] = u[0, i]

            # Forward sweep in j (j = 1 .. n-2)
            for j in seq(1, n - 1):
                p[j, i] = -d / (e + f * p[j - 1, i])
                q[j, i] = (
                    -a * v[j, i - 1]
                    + (1.0 + 2.0 * a) * v[j, i]
                    - c * v[j, i + 1]
                    - f * q[j - 1, i]
                ) / (e + f * p[j - 1, i])

            u[n - 1, i] = 1.0

            # Backward substitution in j (equivalent to j = n-2 .. 0)
            for j in seq(0, n - 1):
                u[n - 2 - j, i] = (
                    p[n - 2 - j, i] * u[n - 1 - j, i]
                    + q[n - 2 - j, i]
                )

# ---------------------------------------------------------------------------
# Scheduling: expose parallelism and improve inner-loop structure.
# ---------------------------------------------------------------------------

kernel_adi_sched = kernel_adi_base

# 1) Parallelize the column-sweep over i.
#    Each iteration updates a distinct row of v, p, q and only reads u.
col_i = kernel_adi_sched.find_loop("i")
kernel_adi_sched = parallelize_loop(kernel_adi_sched, col_i)

# 2) Parallelize the row-sweep over i.
#    Each iteration updates a distinct column of u, p, q and only reads v.
row_i = kernel_adi_sched.find_loop("i #1")
kernel_adi_sched = parallelize_loop(kernel_adi_sched, row_i)

# 3) Tile the forward-sweep j loops to create a fixed-size inner loop that
#    is friendlier to the C compiler's vectorizer (better ILP, fewer branches).
tile_size = 32  # works for arbitrary n; tails are handled safely.

# Column-sweep forward j-loop (first j-loop).
j_col_fwd = kernel_adi_sched.find_loop("j")
kernel_adi_sched = divide_loop(kernel_adi_sched, j_col_fwd, tile_size, ("jo_c", "ji_c"))

# Row-sweep forward j-loop (second remaining j-loop; the first is the
# column-sweep backward-substitution loop, which we leave untouched).
j_row_fwd = kernel_adi_sched.find_loop("j #2")
kernel_adi_sched = divide_loop(kernel_adi_sched, j_row_fwd, tile_size, ("jo_r", "ji_r"))

# Final exported kernel used by the C driver.
kernel_adi = rename(kernel_adi_sched, "kernel_adi")
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