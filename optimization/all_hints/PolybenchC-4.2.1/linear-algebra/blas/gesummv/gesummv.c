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
      /* For 0 <= i < n we have (i % n) == i, so this is equivalent
       * to the original expression but keeps the code identical here
       * to avoid any change in floating-point behaviour.
       */
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
   including the call and return. */
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
  int i;

#pragma scop
  /* Optimized implementation of:
   *
   *   tmp[i] = sum_j A[i][j] * x[j]
   *   y[i]   = alpha * tmp[i] + beta * sum_j B[i][j] * x[j]
   *
   * The original code updated tmp[i] and y[i] on every inner-loop
   * iteration.  Here we:
   *
   *   - Use scalar accumulators (sumA, sumB) so the partial sums stay
   *     in registers instead of repeatedly loading/storing tmp[i] and
   *     y[i].
   *   - Create local 'restrict' views of the arrays to help the
   *     compiler assume no aliasing and generate better code.
   *   - Unroll the inner loop by a factor of 4 to increase ILP and
   *     aid vectorization.
   *   - Optionally parallelize the outer loop with OpenMP.  If the
   *     code is compiled without -fopenmp, the pragma is ignored.
   *
   * The order of operations within each dot product (over j) remains
   * sequential from j = 0 to j = _PB_N-1, so the mathematical
   * behaviour is preserved.
   */

  /* Local, restrict-qualified views to improve alias analysis. */
  DATA_TYPE (* __restrict A1)[n] = A;
  DATA_TYPE (* __restrict B1)[n] = B;
  DATA_TYPE * __restrict tmp1   = tmp;
  DATA_TYPE * __restrict x1     = x;
  DATA_TYPE * __restrict y1     = y;

  const int nPB = _PB_N;

  /* Parallelize across rows.  Each iteration i is independent:
   * it reads A[i][:], B[i][:], x[:] and writes tmp[i], y[i].
   */
#pragma omp parallel for schedule(static) if (nPB > 128)
  for (i = 0; i < nPB; i++)
    {
      DATA_TYPE sumA = SCALAR_VAL(0.0);
      DATA_TYPE sumB = SCALAR_VAL(0.0);

      const DATA_TYPE * __restrict Ai = A1[i];
      const DATA_TYPE * __restrict Bi = B1[i];

      int j = 0;
      /* Largest multiple of 4 not greater than nPB,
       * used for the unrolled main loop.
       */
      const int j_unroll = nPB & ~3;

      /* Unrolled inner loop over j (factor 4).
       * This keeps the dot-products in registers and
       * is friendly to auto-vectorization.
       */
      for (; j < j_unroll; j += 4)
	{
	  const DATA_TYPE x0 = x1[j];
	  const DATA_TYPE x1v = x1[j+1];
	  const DATA_TYPE x2 = x1[j+2];
	  const DATA_TYPE x3 = x1[j+3];

	  sumA += Ai[j]   * x0;
	  sumB += Bi[j]   * x0;

	  sumA += Ai[j+1] * x1v;
	  sumB += Bi[j+1] * x1v;

	  sumA += Ai[j+2] * x2;
	  sumB += Bi[j+2] * x2;

	  sumA += Ai[j+3] * x3;
	  sumB += Bi[j+3] * x3;
	}

      /* Handle remaining elements (at most 3). */
      for (; j < nPB; ++j)
	{
	  const DATA_TYPE xj = x1[j];
	  sumA += Ai[j] * xj;
	  sumB += Bi[j] * xj;
	}

      /* Store results exactly as in the original kernel:
       *
       *   tmp[i] = sum_j A[i][j] * x[j]
       *   y[i]   = alpha * tmp[i] + beta * sum_j B[i][j] * x[j]
       */
      tmp1[i] = sumA;

      DATA_TYPE yi = beta * sumB;
      yi += alpha * sumA;
      y1[i] = yi;
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