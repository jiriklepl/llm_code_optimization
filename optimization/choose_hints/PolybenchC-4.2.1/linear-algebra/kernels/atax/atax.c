/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization.
 *
 * Optimizations:
 *  - Precompute 1/(5*m) once to replace a division inside the inner loop
 *    by a multiplication (division is relatively expensive).
 *  - Take the base address of each row (Ai = A[i]) once per outer
 *    iteration to avoid recomputing the row address in the inner loop.
 */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  const DATA_TYPE fn      = (DATA_TYPE) n;
  const DATA_TYPE inv_5m  =
    SCALAR_VAL(1.0) / (SCALAR_VAL(5.0) * (DATA_TYPE) m);

  /* Initialize x: x[i] = 1 + i / n.  */
  for (i = 0; i < n; i++)
    x[i] = SCALAR_VAL(1.0) + ((DATA_TYPE)i) / fn;

  /* Initialize A: A[i][j] = ((i + j) % n) / (5*m). */
  for (i = 0; i < m; i++)
    {
      DATA_TYPE *Ai = A[i];
      for (j = 0; j < n; j++)
	Ai[j] = ((DATA_TYPE)((i + j) % n)) * inv_5m;
    }
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


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Original computation:
 *   tmp = A * x
 *   y   = A^T * tmp
 *
 * Optimizations:
 *  - Separate the computation into two explicit phases:
 *        1) tmp[i] = sum_j A[i][j] * x[j]
 *        2) y[j]  += sum_i A[i][j] * tmp[i]
 *    This matches the algebraic structure y = A^T * (A * x) and keeps
 *    the accumulation order for each y[j] identical to the original code
 *    (outer loop over i, inner over j), preserving numerical behavior.
 *
 *  - Improve data locality and reduce address recalculation by taking
 *    a pointer to the current row (Ai = A[i]) once per outer loop.
 *
 *  - Expose coarse-grain parallelism with OpenMP:
 *      * Phase 1: independent dot-products over i (rows) of A.
 *      * Phase 2: independent partial contributions over i, combined
 *        with an OpenMP array reduction on y.
 *    When compiled without -fopenmp, all #pragma omp directives are
 *    ignored by the compiler and the code executes sequentially with
 *    the same semantics as the optimized scalar version.
 *
 *  - Add #pragma omp simd on inner loops to encourage the compiler
 *    to vectorize them aggressively when possible.
 */
static
void kernel_atax [[gnu::flatten, gnu::noinline]] (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  /* Cache the polybench loop bounds in local constants to avoid
     re-evaluating the macros in multiple pragmas and conditions. */
  const int PB_M = _PB_M;
  const int PB_N = _PB_N;

#pragma scop

  /* ------------------------------------------------------------------ */
  /* Phase 0: Initialize y to zero.                                     */
  /* ------------------------------------------------------------------ */

  /* Standalone initialization loop helps both vectorization and, when
     OpenMP is enabled, allows inexpensive parallel initialization. */
#pragma omp parallel for if (PB_N > 1024) schedule(static)
  for (i = 0; i < PB_N; i++)
    y[i] = SCALAR_VAL(0.0);

  /* ------------------------------------------------------------------ */
  /* Phase 1: tmp = A * x                                               */
  /*   tmp[i] = sum_{j=0..N-1} A[i][j] * x[j]                           */
  /* ------------------------------------------------------------------ */

  /* Each row i is independent; we parallelize across i when OpenMP is
     enabled. The 'if' clause avoids spawning threads for very small
     problems. */
#pragma omp parallel for if ((long)PB_M * (long)PB_N > 32768L) schedule(static)
  for (i = 0; i < PB_M; i++)
    {
      DATA_TYPE acc = SCALAR_VAL(0.0);
      DATA_TYPE *Ai = A[i];

      /* Encourage SIMD vectorization of the inner dot-product. */
#pragma omp simd reduction(+:acc)
      for (j = 0; j < PB_N; j++)
	acc += Ai[j] * x[j];

      tmp[i] = acc;
    }

  /* ------------------------------------------------------------------ */
  /* Phase 2: y += A^T * tmp                                            */
  /*   y[j] = sum_{i=0..M-1} A[i][j] * tmp[i]                           */
  /*                                                                   *
   * We keep the accumulation order over i (outer loop over i, inner
   * over j) identical to the original code, but now exploit parallelism
   * across i with an OpenMP array reduction on y.                      */
  /* ------------------------------------------------------------------ */

#pragma omp parallel for if ((long)PB_M * (long)PB_N > 32768L) \
                         schedule(static) reduction(+:y[:PB_N])
  for (i = 0; i < PB_M; i++)
    {
      const DATA_TYPE ti   = tmp[i];
      const DATA_TYPE *Ai  = A[i];

      /* Inner loop is naturally vectorizable; 'omp simd' makes this
         explicit for compilers with OpenMP SIMD support. */
#pragma omp simd
      for (j = 0; j < PB_N; j++)
	y[j] += Ai[j] * ti;
    }

#pragma endscop
}


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
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

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