/**
 * Exo SYR2K driver: mirrors PolyBench/C syr2k.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"

/* Include the Exo-generated kernel header. */
#include "generated/syr2k/syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
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

# Base SYR2K kernel operating on the lower-triangular part of C.
@proc
def kernel_syr2k_base(
    n: size,
    m: size,
    alpha: DATA_TYPE,
    beta: DATA_TYPE,
    C: DATA_TYPE[n, n] @ DRAM,
    A: DATA_TYPE[n, m] @ DRAM,
    B: DATA_TYPE[n, m] @ DRAM,
):
    # Basic sanity assertions on problem sizes.
    assert n >= 0
    assert m >= 0

    for i in seq(0, n):
        # Scale the lower-triangular part of row i of C by beta.
        for j in seq(0, i + 1):
            C[i, j] = beta * C[i, j]

        # Rank-2k update for row i.
        for k in seq(0, m):
            # Temporaries holding alpha * A[i, k] and alpha * B[i, k].
            # These will be reused across all j for a fixed (i, k).
            aik: DATA_TYPE
            bik: DATA_TYPE

            aik = alpha * A[i, k]
            bik = alpha * B[i, k]

            for j in seq(0, i + 1):
                # C[i, j] += alpha * (A[j, k] * B[i, k] + B[j, k] * A[i, k])
                # Implemented as:
                #   A[j, k] * (alpha * B[i, k]) + B[j, k] * (alpha * A[i, k])
                C[i, j] += A[j, k] * bik + B[j, k] * aik


def schedule_syr2k(p):
    # Lift the temporaries so they are allocated once per i-iteration
    # instead of once per (i, k) iteration.
    aik_alloc = p.find("aik : _")
    p = lift_alloc(p, aik_alloc)

    bik_alloc = p.find("bik : _")
    p = lift_alloc(p, bik_alloc)

    # Parallelize the outer loop over i. Each iteration updates a distinct
    # row of the lower-triangular part of C, so there are no dependencies
    # between different i-iterations.
    i_loop = p.find_loop("i")
    p = parallelize_loop(p, i_loop)

    # Simplify the IR after the scheduling transformations to clean up
    # any redundant structure introduced during rewriting.
    p = simplify(p)

    # Rename the scheduled procedure to match the C entry point expected
    # by the driver.
    p = rename(p, "kernel_syr2k")
    return p


# Final optimized kernel exposed to the C driver.
kernel_syr2k = schedule_syr2k(kernel_syr2k_base)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_syr2k (/*ctxt=*/NULL, n, m,
		(DATA_TYPE*)&alpha, (DATA_TYPE*)&beta,
		(DATA_TYPE*)POLYBENCH_ARRAY(C),
		(DATA_TYPE*)POLYBENCH_ARRAY(A),
		(DATA_TYPE*)POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}