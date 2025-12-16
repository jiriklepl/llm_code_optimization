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
    assert n > 2

    a: DATA_TYPE
    b: DATA_TYPE
    c: DATA_TYPE
    d: DATA_TYPE
    e: DATA_TYPE
    f: DATA_TYPE

    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in seq(1, tsteps + 1):
        # Column sweep
        for i in seq(1, n - 1):
            v[0, i] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = v[0, i]

            for j in seq(1, n - 1):
                p[i, j] = -c / (a * p[i, j - 1] + b)
                q[i, j] = (
                    -d * u[j, i - 1]
                    + (1.0 + 2.0 * d) * u[j, i]
                    - f * u[j, i + 1]
                    - a * q[i, j - 1]
                ) / (a * p[i, j - 1] + b)

            v[n - 1, i] = 1.0
            for j in seq(1, n - 1):
                v[n - 1 - j, i] = (
                    p[i, n - 1 - j] * v[n - j, i]
                    + q[i, n - 1 - j]
                )

        # Row sweep
        for i in seq(1, n - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = u[i, 0]

            for j in seq(1, n - 1):
                p[i, j] = -f / (d * p[i, j - 1] + e)
                q[i, j] = (
                    -a * v[i - 1, j]
                    + (1.0 + 2.0 * a) * v[i, j]
                    - c * v[i + 1, j]
                    - d * q[i, j - 1]
                ) / (d * p[i, j - 1] + e)

            u[i, n - 1] = 1.0
            for j in seq(1, n - 1):
                u[i, n - 1 - j] = (
                    p[i, n - 1 - j] * u[i, n - j]
                    + q[i, n - 1 - j]
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