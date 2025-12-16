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
        A[i][j] = (DATA_TYPE) (i*j % n) / fn;
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

  /* Local const copies help the compiler keep these in registers. */
  const DATA_TYPE alpha_ = alpha;
  const DATA_TYPE beta_  = beta;

#pragma scop

  /* --------------------------------------------------------------------
   * 1) Rank-2 update: A := A + u1*v1^T + u2*v2^T
   *
   *    We keep the original loop order (i outer, j inner), which already
   *    traverses A row-wise (contiguous in memory).  We hoist u1[i] and
   *    u2[i] out of the inner loop and take a row pointer to improve
   *    register reuse and reduce address calculations.
   * ------------------------------------------------------------------ */
  for (i = 0; i < _PB_N; i++)
  {
    const DATA_TYPE u1_i = u1[i];
    const DATA_TYPE u2_i = u2[i];
    DATA_TYPE *restrict A_row = A[i];

    for (j = 0; j < _PB_N; j++)
    {
      A_row[j] += u1_i * v1[j] + u2_i * v2[j];
    }
  }

  /* --------------------------------------------------------------------
   * 2) x := x + beta * A^T * y
   *
   * Original code:
   *   for (i)
   *     for (j)
   *       x[i] += beta * A[j][i] * y[j];
   *
   * This walks A by columns (A[j][i]) with a large stride, which is
   * cache-unfriendly.  We interchange the loops so that the inner loop
   * walks an entire row of A contiguously:
   *
   *   for (j)
   *     for (i)
   *       x[i] += beta * y[j] * A[j][i];
   *
   * For each fixed i, the updates to x[i] still occur in increasing j
   * order (0.._PB_N-1), exactly as in the original program, so the
   * floating-point accumulation order for each element of x is preserved.
   * Only updates to different elements of x are interleaved, which is
   * allowed since they are independent.
   * ------------------------------------------------------------------ */
  for (j = 0; j < _PB_N; j++)
  {
    const DATA_TYPE y_j      = y[j];
    const DATA_TYPE scale_j  = beta_ * y_j;
    DATA_TYPE *restrict A_row = A[j];  /* row j of A */

    for (i = 0; i < _PB_N; i++)
    {
      x[i] += scale_j * A_row[i];
    }
  }

  /* --------------------------------------------------------------------
   * 3) x := x + z
   *    Simple element-wise update; left unchanged except for using +=.
   * ------------------------------------------------------------------ */
  for (i = 0; i < _PB_N; i++)
    x[i] += z[i];

  /* --------------------------------------------------------------------
   * 4) w := w + alpha * A * x
   *
   * Original code:
   *   for (i)
   *     for (j)
   *       w[i] += alpha * A[i][j] * x[j];
   *
   * We keep the original loop order (row-wise traversal of A), but we
   * accumulate into a local scalar 'wi' to avoid repeatedly loading and
   * storing w[i] from memory.  The sequence of additions applied to
   * w[i] is unchanged; only the temporary storage location differs.
   * ------------------------------------------------------------------ */
  for (i = 0; i < _PB_N; i++)
  {
    DATA_TYPE *restrict A_row = A[i];
    DATA_TYPE wi = w[i];

    for (j = 0; j < _PB_N; j++)
    {
      wi += alpha_ * A_row[j] * x[j];
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