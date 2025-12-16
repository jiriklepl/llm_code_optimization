/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  /* Tunable tile size for the inner dimension of the final GEMV.
     A small power-of-two that fits well in L1/L2 is a good default. */
  const int TILE_COL = 64;

#pragma scop

  /*
   * ------------------------------------------------------------------
   * Phase 1: Fused rank-2 update of A and x := x + beta * A^T * y
   * ------------------------------------------------------------------
   *
   * Original code (first two loops):
   *
   *   // 1) A := A + u1 v1^T + u2 v2^T
   *   for (i = 0; i < _PB_N; i++)
   *     for (j = 0; j < _PB_N; j++)
   *       A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
   *
   *   // 2) x := x + beta * A^T * y
   *   for (i = 0; i < _PB_N; i++)
   *     for (j = 0; j < _PB_N; j++)
   *       x[i] = x[i] + beta * A[j][i] * y[j];
   *
   * After phase 1 the mathematical effect is:
   *   A' = A + u1 v1^T + u2 v2^T
   *   x  = x + beta * A'^T y
   *
   * The transformation below:
   *   - processes A row by row (j is the row index),
   *   - updates A[j][*] using u1[j] and u2[j],
   *   - immediately uses the updated A[j][*] to accumulate into x[*].
   *
   * This is equivalent to the two original loops, but:
   *   - touches each element of A only once in this phase,
   *   - accesses A[j][i], v1[i], v2[i], and x[i] with unit stride,
   *   - hoists beta * y[j] and u1[j], u2[j] out of the inner loop.
   */

  for (j = 0; j < _PB_N; ++j)
  {
    DATA_TYPE u1_j    = u1[j];
    DATA_TYPE u2_j    = u2[j];
    DATA_TYPE beta_yj = beta * y[j];

    for (i = 0; i < _PB_N; ++i)
    {
      /* Update A[j][i] (rank-2 update) */
      DATA_TYPE a_ji = A[j][i] + u1_j * v1[i] + u2_j * v2[i];
      A[j][i] = a_ji;

      /* Accumulate contribution of row j of A' into x[i]:
         x[i] += beta * A'[j][i] * y[j] = beta_yj * a_ji; */
      x[i] += beta_yj * a_ji;
    }
  }

  /*
   * ------------------------------------------------------------------
   * Phase 2: x := x + z
   * ------------------------------------------------------------------
   *
   * Original code (third loop):
   *
   *   for (i = 0; i < _PB_N; i++)
   *     x[i] = x[i] + z[i];
   *
   * We keep it as a separate, simple vectorizable loop.
   * Final x is:
   *   x = x_initial + beta * A'^T y + z
   * which matches the original computation.
   */

  for (i = 0; i < _PB_N; ++i)
  {
    x[i] += z[i];
  }

  /*
   * ------------------------------------------------------------------
   * Phase 3: w := w + alpha * A * x
   * ------------------------------------------------------------------
   *
   * Original code (fourth loop):
   *
   *   for (i = 0; i < _PB_N; i++)
   *     for (j = 0; j < _PB_N; j++)
   *       w[i] = w[i] + alpha * A[i][j] * x[j];
   *
   * We keep the natural row-major traversal (i outer, j inner) but:
   *   - use a per-row accumulator 'wi' to limit loads/stores of w[i],
   *   - block the j-loop to improve locality for large N, so that
   *     chunks of x[j] and A[i][j] stay in cache.
   */

  for (i = 0; i < _PB_N; ++i)
  {
    DATA_TYPE wi = w[i];

    for (int jj = 0; jj < _PB_N; jj += TILE_COL)
    {
      int j_end = jj + TILE_COL;
      if (j_end > _PB_N)
        j_end = _PB_N;

      for (j = jj; j < j_end; ++j)
      {
        wi += alpha * A[i][j] * x[j];
      }
    }

    w[i] = wi;
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
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}