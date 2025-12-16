/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"

/* --------------------------------------------------------------------
 * Simple tuning knobs for the kernel implementation.
 *
 * These macros can be overridden at compile time, e.g.:
 *   -DGESUMMV_OMP_MIN_N=1024 -DGESUMMV_OMP_SCHEDULE=dynamic
 *
 * GESUMMV_OMP_MIN_N:
 *   Minimum problem size above which OpenMP parallelization of
 *   the outer loop is enabled (when compiled with -fopenmp).
 *
 * GESUMMV_OMP_SCHEDULE:
 *   OpenMP schedule policy used for the outer loop.
 *   Typical values: static, dynamic, guided.
 * ------------------------------------------------------------------ */
#ifndef GESUMMV_OMP_MIN_N
#define GESUMMV_OMP_MIN_N 256
#endif

#ifndef GESUMMV_OMP_SCHEDULE
#define GESUMMV_OMP_SCHEDULE static
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
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

   Optimizations applied:
   - Use alias-restricted local pointers to help the compiler's
     alias analysis and enable more aggressive auto-vectorization.
   - Accumulate the dot products in scalar temporaries (tmp_i, y_i)
     instead of repeatedly reading/writing tmp[i] and y[i] inside
     the inner loop. This reduces memory traffic and improves
     cache utilization.
   - Optionally parallelize the outer i-loop with OpenMP. Each
     iteration is independent, so this is safe. When compiled
     without -fopenmp, the pragma is ignored and the code executes
     sequentially as in the original version.

   The mathematical computation and the per-element evaluation order
   of the inner loop are preserved for each row i.
*/
static
void kernel_gesummv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  /* Local alias-restricted views of the arrays.
   *
   * The PolyBench macros expand A and B to variable-length 2D arrays
   * with second dimension 'n'.  Taking them through restrict-qualified
   * pointers tells the compiler that:
   *   - A, B, x, y, tmp do not overlap in memory
   *   - Each of these pointers refers to a contiguous block
   *
   * This significantly helps auto-vectorization and instruction
   * scheduling without changing the observable behavior.
   */
  DATA_TYPE (* __restrict A_restrict)[n] = A;
  DATA_TYPE (* __restrict B_restrict)[n] = B;
  DATA_TYPE * __restrict tmp_restrict    = tmp;
  DATA_TYPE * __restrict x_restrict      = x;
  DATA_TYPE * __restrict y_restrict      = y;

  const int n_eff = _PB_N; /* effective problem size used in loops */
  int i;

#pragma scop
  /* Parallelize across rows when OpenMP is enabled and the problem
   * is large enough.  The 'if' clause lets the runtime decide
   * between parallel and serial execution, avoiding parallel overhead
   * on very small problems.  When compiled without -fopenmp this
   * pragma is ignored and the loop executes sequentially. */
#pragma omp parallel for schedule(GESUMMV_OMP_SCHEDULE) if (n_eff >= GESUMMV_OMP_MIN_N)
  for (i = 0; i < n_eff; i++)
    {
      /* Per-row accumulators kept in registers.  This preserves the
       * per-row evaluation order of the original inner loop:
       *   tmp_i = (((0 + v0) + v1) + v2) + ...
       * and similarly for y_i. */
      DATA_TYPE tmp_i = SCALAR_VAL(0.0);
      DATA_TYPE y_i   = SCALAR_VAL(0.0);

      const DATA_TYPE * __restrict Ai = A_restrict[i];
      const DATA_TYPE * __restrict Bi = B_restrict[i];

      int j;
      for (j = 0; j < n_eff; j++)
	{
	  const DATA_TYPE xj = x_restrict[j];
	  tmp_i += Ai[j] * xj;
	  y_i   += Bi[j] * xj;
	}

      /* Write the final results back to memory once per row. */
      tmp_restrict[i] = tmp_i;
      y_restrict[i]   = alpha * tmp_i + beta * y_i;
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gesummv (n, alpha, beta,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(B),
		  POLYBENCH_ARRAY(tmp),
		  POLYBENCH_ARRAY(x),
		  POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}