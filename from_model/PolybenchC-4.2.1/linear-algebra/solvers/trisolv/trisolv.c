/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations applied:
 *   - Use a scalar accumulator for each row (express the inner loop as a
 *     reduction) to keep x[i] in a register until the end of the row.
 *   - Introduce local restrict-qualified pointers to L, x, and b to help
 *     the compiler with alias analysis and enable better vectorization.
 *   - Manually unroll the inner j-loop by a small factor (4) to increase
 *     instruction-level parallelism and reduce loop overhead, while
 *     preserving the original mathematical computation:
 *
 *       x[i] = ( b[i] - sum_{j=0}^{i-1} L[i][j] * x[j] ) / L[i][i];
 *
 *   - Keep the outer loop on i strictly sequential to respect the
 *     forward-substitution dependencies.
 */
static
void kernel_trisolv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  /* Create local restrict-qualified views to help the optimizer.
   * The PolyBench framework allocates L, x, and b in disjoint regions,
   * so this is semantically correct and allows more aggressive
   * vectorization and unrolling.
   */
  DATA_TYPE (*restrict Lp)[n] = L;
  DATA_TYPE *restrict xp      = x;
  DATA_TYPE *restrict bp      = b;

  /* Unroll factor for the inner j loop.
   * Chosen as a small power of two for good balance between
   * code size and ILP. This can be tuned if desired.
   */
  const int unroll_j = 4;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* Start from the right-hand side value for row i. */
      DATA_TYPE sum = bp[i];

      /* Cache the base address of row i of L to avoid repeated
       * pointer arithmetic inside the inner loops.
       */
      DATA_TYPE *restrict Li = Lp[i];

      /* Manually unrolled inner product:
       *
       *   sum = b[i] - sum_{j=0}^{i-1} L[i][j] * x[j]
       *
       * First handle the largest multiple of unroll_j less than i
       * using an unrolled loop, then clean up the remaining elements.
       */
      int j_end = i - (i % unroll_j);

      /* Unrolled main part of the reduction over j.
       * Accesses:
       *   - Li[j]   ... Li[j+3]      are contiguous in memory (row-major),
       *   - xp[j]   ... xp[j+3]      are also contiguous.
       * This pattern is well-suited for SIMD code generation.
       */
      for (j = 0; j < j_end; j += unroll_j)
        {
          sum -= Li[j]   * xp[j];
          sum -= Li[j+1] * xp[j+1];
          sum -= Li[j+2] * xp[j+2];
          sum -= Li[j+3] * xp[j+3];
        }

      /* Remainder loop for the last (i % unroll_j) elements, if any. */
      for (; j < i; j++)
        {
          sum -= Li[j] * xp[j];
        }

      /* Final division by the diagonal element.
       * This matches the original:
       *   x[i] = x[i] / L[i][i];
       * but we kept x[i] in a register (sum) throughout the inner loop.
       */
      xp[i] = sum / Li[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}