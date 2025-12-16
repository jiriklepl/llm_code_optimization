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
#include <omp.h>  /* Added for OpenMP parallelization and SIMD hints. */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization.
 * Slightly reorganized to reduce divisions and improve memory access
 * while preserving the original initialization semantics.
 */
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
  *beta  = 1.2;

  const DATA_TYPE fn     = (DATA_TYPE)n;
  const DATA_TYPE inv_fn = (DATA_TYPE)1.0 / fn;

  const DATA_TYPE inv_2  = (DATA_TYPE)0.5;          /* 1/2  */
  const DATA_TYPE inv_4  = (DATA_TYPE)0.25;         /* 1/4  */
  const DATA_TYPE inv_6  = (DATA_TYPE)(1.0 / 6.0);  /* 1/6  */
  const DATA_TYPE inv_8  = (DATA_TYPE)0.125;        /* 1/8  */
  const DATA_TYPE inv_9  = (DATA_TYPE)(1.0 / 9.0);  /* 1/9  */

  /* 1/n factor for A[i][j] initialization. */
  const DATA_TYPE inv_n  = inv_fn;

  for (i = 0; i < n; i++)
    {
      DATA_TYPE ratio = ((DATA_TYPE)(i + 1)) * inv_fn;

      u1[i] = (DATA_TYPE)i;
      u2[i] = ratio * inv_2;
      v1[i] = ratio * inv_4;
      v2[i] = ratio * inv_6;
      y[i]  = ratio * inv_8;
      z[i]  = ratio * inv_9;
      x[i]  = (DATA_TYPE)0.0;
      w[i]  = (DATA_TYPE)0.0;

      /* Work with a row pointer for better locality. */
      DATA_TYPE *Ai = A[i];

      for (j = 0; j < n; j++)
        Ai[j] = ((DATA_TYPE)((i * j) % n)) * inv_n;
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
   including the call and return.
 *
 * Optimizations applied:
 *  - Use restrict-qualified local aliases to aid vectorization.
 *  - Parallelize outer loops with OpenMP.
 *  - Reorder the GEMV with A^T to iterate over rows of A (better cache use)
 *    and use an OpenMP array reduction for x.
 *  - Express the final GEMV (w update) as a dot product per row to remove
 *    one multiplication by alpha per inner iteration.
 *  - Add OpenMP SIMD hints on inner loops.
 */
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

  /* Effective problem size (may differ from N through PolyBench macros). */
  const int n_eff = _PB_N;

  /* Local restrict-qualified aliases to help the compiler with
     alias analysis and vectorization. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE *restrict u1_ = u1;
  DATA_TYPE *restrict v1_ = v1;
  DATA_TYPE *restrict u2_ = u2;
  DATA_TYPE *restrict v2_ = v2;
  DATA_TYPE *restrict w_  = w;
  DATA_TYPE *restrict x_  = x;
  DATA_TYPE *restrict y_  = y;
  DATA_TYPE *restrict z_  = z;

#pragma scop

  /* --------------------------------------------------------------
   * 1) Rank-1 updates: A = A + u1 * v1^T + u2 * v2^T
   *    Parallel over rows; inner loop is contiguous in memory.
   * -------------------------------------------------------------- */
#pragma omp parallel for private(j) schedule(static)
  for (i = 0; i < n_eff; i++)
    {
      DATA_TYPE *restrict Ai = A_[i];
      const DATA_TYPE u1_i = u1_[i];
      const DATA_TYPE u2_i = u2_[i];

#pragma omp simd
      for (j = 0; j < n_eff; j++)
        {
          Ai[j] += u1_i * v1_[j] + u2_i * v2_[j];
        }
    }

  /* --------------------------------------------------------------
   * 2) x = x + beta * A^T * y
   *
   * Original code:
   *   for (i)
   *     for (j)
   *       x[i] += beta * A[j][i] * y[j];
   *
   * This is mathematically:
   *   x[i] += beta * sum_j (A[j][i] * y[j])
   *
   * We reorder loops to iterate over rows of A (better locality)
   * and use an OpenMP array reduction on x.
   * -------------------------------------------------------------- */
#pragma omp parallel for private(i) schedule(static) reduction(+:x_[0:n_eff])
  for (j = 0; j < n_eff; j++)
    {
      const DATA_TYPE *restrict Aj = A_[j]; /* Row j: Aj[i] == A[j][i] */
      const DATA_TYPE beta_yj = beta * y_[j];

#pragma omp simd
      for (i = 0; i < n_eff; i++)
        {
          x_[i] += Aj[i] * beta_yj;
        }
    }

  /* --------------------------------------------------------------
   * 3) x = x + z
   *    Simple element-wise update, parallelized.
   * -------------------------------------------------------------- */
#pragma omp parallel for schedule(static)
  for (i = 0; i < n_eff; i++)
    {
      x_[i] += z_[i];
    }

  /* --------------------------------------------------------------
   * 4) w = w + alpha * A * x
   *
   * Original code:
   *   for (i)
   *     for (j)
   *       w[i] += alpha * A[i][j] * x[j];
   *
   * Mathematically:
   *   w[i] = w[i] + alpha * sum_j (A[i][j] * x[j])
   *
   * We compute a dot product per row, factorizing alpha outside
   * the inner loop to reduce the number of multiplications.
   * -------------------------------------------------------------- */
#pragma omp parallel for private(j) schedule(static)
  for (i = 0; i < n_eff; i++)
    {
      const DATA_TYPE *restrict Ai = A_[i];
      DATA_TYPE dot = (DATA_TYPE)0.0;

#pragma omp simd reduction(+:dot)
      for (j = 0; j < n_eff; j++)
        {
          dot += Ai[j] * x_[j];
        }

      w_[i] += alpha * dot;
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