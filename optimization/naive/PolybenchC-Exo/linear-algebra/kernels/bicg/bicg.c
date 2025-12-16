/**
 * Exo BiCG driver: mirrors PolyBench/C bicg.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"

/* Include the Exo-generated kernel header. */
#include "generated/bicg/bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# BiCG kernel:
#   s[j] = sum_{i=0}^{n-1} r[i] * A[i, j]
#   q[i] = sum_{j=0}^{m-1} A[i, j] * p[j]
@proc
def kernel_bicg(
    m: size,
    n: size,
    A: DATA_TYPE[n, m] @ DRAM,
    s: DATA_TYPE[m] @ DRAM,
    q: DATA_TYPE[n] @ DRAM,
    p: DATA_TYPE[m] @ DRAM,
    r: DATA_TYPE[n] @ DRAM,
):
    # Initialize s to zero over its m entries
    for j in seq(0, m):
        s[j] = 0.0

    # Main BiCG computation: one row of A at a time (row-major friendly).
    for i in seq(0, n):
        q[i] = 0.0
        for j in seq(0, m):
            # Fused update of s and q to reuse A[i, j] in caches.
            s[j] += r[i] * A[i, j]
            q[i] += A[i, j] * p[j]


# -----------------------
# Scheduling / optimization
# -----------------------

# Unroll factor for the j dimension. A value of 8 matches common SIMD
# widths (8 floats for AVX2 or 8 doubles for AVX-512) and improves
# instruction-level parallelism while preserving the exact iteration order.
J_TILE = 8

# 1) Tile and unroll the initialization loop over s[j].
#    This turns the simple linear pass over s into blocks of 8
#    contiguous elements, which the C compiler can better unroll/vectorize.
kernel_bicg = divide_loop(kernel_bicg, "j", J_TILE, ("jo_s", "ji_s"), tail="cut")
kernel_bicg = unroll_loop(kernel_bicg, "ji_s")

# 2) Tile and unroll the inner loop over j in the main BiCG computation.
#    The transformation keeps the overall execution order of j intact,
#    but exposes 8-way blocks of contiguous accesses in A, s, and p
#    for each fixed i, which is cache- and SIMD-friendly.
kernel_bicg = divide_loop(kernel_bicg, "j", J_TILE, ("jo", "ji"), tail="cut")
kernel_bicg = unroll_loop(kernel_bicg, "ji")

# 3) Simplify the resulting code to clean up index arithmetic and
#    constant expressions introduced by tiling/unrolling.
kernel_bicg = simplify(kernel_bicg)
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_bicg(/*ctxt=*/NULL, m, n,
	      (DATA_TYPE*)POLYBENCH_ARRAY(A),
	      (DATA_TYPE*)POLYBENCH_ARRAY(s),
	      (DATA_TYPE*)POLYBENCH_ARRAY(q),
	      (DATA_TYPE*)POLYBENCH_ARRAY(p),
	      (DATA_TYPE*)POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}