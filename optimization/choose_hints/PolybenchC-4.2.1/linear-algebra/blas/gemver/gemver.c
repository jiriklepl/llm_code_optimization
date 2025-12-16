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

#pragma scop

  /* Use a local copy of the loop bound for readability. */
  const int n_pb = _PB_N;

  /* Create local restrict-qualified aliases to help the compiler with
   * alias analysis and vectorization. The PolyBench argument macros
   * do not necessarily use 'restrict', and these aliases are the only
   * ones used below to access the data.
   */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE *restrict u1_ = u1;
  DATA_TYPE *restrict v1_ = v1;
  DATA_TYPE *restrict u2_ = u2;
  DATA_TYPE *restrict v2_ = v2;
  DATA_TYPE *restrict w_  = w;
  DATA_TYPE *restrict x_  = x;
  DATA_TYPE *restrict y_  = y;
  DATA_TYPE *restrict z_  = z;

  /* ------------------------------------------------------------------
   * Fused update of A and computation of x
   *
   * Original code:
   *
   *   for (i)
   *     for (j)
   *       A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
   *
   *   for (i)
   *     for (j)
   *       x[i] = x[i] + beta * A[j][i] * y[j];
   *
   * After the first loop we have:
   *   A'[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
   *
   * After the second loop:
   *   x[i]_out = x[i]_in + beta * sum_j A'[j][i] * y[j].
   *
   * In the fused version we iterate by rows i, compute the updated
   * A'[i][j], and immediately accumulate that row's contribution to
   * x[j]:
   *
   *   x[j] += beta * y[i] * A'[i][j]  for all i,j
   *
   * which is mathematically identical, since
   *   sum_j A'[j][i] * y[j]  ==  sum_j A'[j][i] * y[j]
   * and we just change the order of summation.
   *
   * This fusion:
   *   - touches each A_[i][j] only once in this phase,
   *   - keeps accesses to A_ row-major (good locality),
   *   - avoids a separate strided pass over A_ for the x update.
   *
   * The OpenMP reduction on x_ is optional: with -fopenmp it enables
   * parallel execution over i while safely accumulating into x_;
   * without OpenMP support the pragma is ignored and the loop executes
   * sequentially with the same semantics as the fused version.
   * ------------------------------------------------------------------ */
#pragma omp parallel for reduction(+:x_[:n_pb]) private(j)
  for (i = 0; i < n_pb; i++)
  {
    DATA_TYPE *restrict Ai = A_[i];
    const DATA_TYPE u1_i = u1_[i];
    const DATA_TYPE u2_i = u2_[i];
    const DATA_TYPE beta_y_i = beta * y_[i];

    for (j = 0; j < n_pb; j++)
    {
      const DATA_TYPE v1_j = v1_[j];
      const DATA_TYPE v2_j = v2_[j];

      DATA_TYPE a_ij = Ai[j];
      a_ij += u1_i * v1_j + u2_i * v2_j;
      Ai[j] = a_ij;

      x_[j] += beta_y_i * a_ij;
    }
  }

  /* ------------------------------------------------------------------
   * x[i] = x[i] + z[i];
   *
   * This loop is embarrassingly parallel across i.
   * ------------------------------------------------------------------ */
#pragma omp parallel for
  for (i = 0; i < n_pb; i++)
    x_[i] += z_[i];

  /* ------------------------------------------------------------------
   * w[i] = w[i] + alpha * sum_j A[i][j] * x[j];
   *
   * Parallelized across i (rows of A). Each iteration works on a
   * distinct element w_[i] and a distinct row A_[i][:], so there are
   * no cross-iteration dependencies.
   *
   * We keep a row-local accumulator 'wi' in a register to reduce
   * memory traffic on w_.
   * ------------------------------------------------------------------ */
#pragma omp parallel for private(j)
  for (i = 0; i < n_pb; i++)
  {
    DATA_TYPE *restrict Ai = A_[i];
    DATA_TYPE wi = w_[i];

    for (j = 0; j < n_pb; j++)
      wi += alpha * Ai[j] * x_[j];

    w_[i] = wi;
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