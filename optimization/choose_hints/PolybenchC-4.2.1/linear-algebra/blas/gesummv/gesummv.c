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
      x[i] = (DATA_TYPE)(i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE)((i * j + 1) % n) / n;
	B[i][j] = (DATA_TYPE)((i * j + 2) % n) / n;
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
   - Use restrict-qualified local pointers to tell the compiler that
     A, B, x, y, and tmp do not alias, enabling better vectorization.
   - Replace repeated loads/stores of tmp[i] and y[i] inside the
     inner loop with scalar accumulators kept in registers.
   - Keep the access pattern A[i][j], B[i][j], x[j] so that all
     inner-loop accesses are contiguous in memory (good locality).
   - Optionally parallelize the outer loop over i with OpenMP. When
     compiled without OpenMP support, the pragmas are ignored and the
     code runs sequentially, preserving original behavior.
*/
static void __attribute__((flatten, noinline))
kernel_gesummv(int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

  /* Local restrict-qualified aliases: these express to the compiler
   * that the pointed-to regions do not overlap, which can remove
   * unnecessary reloads and enable more aggressive vectorization.
   */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict B_)[n] = B;
  DATA_TYPE *restrict tmp_    = tmp;
  DATA_TYPE *restrict x_      = x;
  DATA_TYPE *restrict y_      = y;

#pragma scop
  /* Each iteration of the outer loop works on a distinct row i and
   * writes to tmp_[i] and y_[i] only, so it can be safely parallelized.
   * If OpenMP is not enabled at compile time, this pragma is ignored.
   */
#pragma omp parallel for private(j) schedule(static)
  for (i = 0; i < _PB_N; i++)
    {
      /* Scalar accumulators for the two dot products corresponding to
       * row i. Keeping them in registers avoids repeated memory
       * traffic on tmp_[i] and y_[i] in the inner loop.
       */
      DATA_TYPE tmp_i = (DATA_TYPE)0;
      DATA_TYPE y_i   = (DATA_TYPE)0;

      /* Compute:
       *   tmp_i = sum_j A[i][j] * x[j]
       *   y_i   = sum_j B[i][j] * x[j]
       *
       * The 'omp simd' directive (ignored unless OpenMP SIMD is enabled)
       * documents that this loop is a pure reduction and can be safely
       * vectorized.
       */
#pragma omp simd reduction(+:tmp_i, y_i)
      for (j = 0; j < _PB_N; j++)
	{
	  const DATA_TYPE xj  = x_[j];
	  const DATA_TYPE Aij = A_[i][j];
	  const DATA_TYPE Bij = B_[i][j];

	  tmp_i += Aij * xj;
	  y_i   += Bij * xj;
	}

      /* Store the full dot-product result in tmp_ (as in the original
       * code) and form the final linear combination for y_.
       */
      tmp_[i] = tmp_i;
      y_[i]   = alpha * tmp_i + beta * y_i;
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