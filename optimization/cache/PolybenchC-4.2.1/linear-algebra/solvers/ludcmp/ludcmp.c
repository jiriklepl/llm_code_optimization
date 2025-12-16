/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

/* Tunable blocking parameters for the Gram-matrix construction
 * used in init_array. They can be overridden at compile time, e.g.:
 *
 *   -DINIT_GRAM_BLOCK_R=64 -DINIT_GRAM_BLOCK_S=64
 *
 * They control the tile sizes in the (r,s) dimensions.
 */
#ifndef INIT_GRAM_BLOCK_R
# define INIT_GRAM_BLOCK_R 32
#endif

#ifndef INIT_GRAM_BLOCK_S
# define INIT_GRAM_BLOCK_S 32
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  /* Initialize vectors. */
  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  /* Initialize A as a lower-triangular matrix with unit diagonal. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *Ai = A[i];

      for (j = 0; j <= i; j++)
	Ai[j] = (DATA_TYPE)(-j % n) / n + 1;

      for (j = i+1; j < n; j++)
	Ai[j] = 0;

      Ai[i] = 1;
    }

  /* Make the matrix positive semi-definite.
   *
   * This computes B = A * A^T in a cache-friendly way.  It is not
   * required for LU itself but is retained for consistency with the
   * original PolyBench/Cholesky setup.
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  DATA_TYPE (*B_)[n] = POLYBENCH_ARRAY(B);

  /* Blocked computation of B[r][s] = sum_t A[r][t] * A[s][t].
   * For each (r,s) pair the accumulation over t is performed in the
   * same order (t = 0..n-1) as in the original implementation, so the
   * floating-point result is bitwise identical.
   */
  for (int rb = 0; rb < n; rb += INIT_GRAM_BLOCK_R)
    {
      int r_end = rb + INIT_GRAM_BLOCK_R;
      if (r_end > n)
        r_end = n;

      for (int sb = 0; sb < n; sb += INIT_GRAM_BLOCK_S)
        {
          int s_end = sb + INIT_GRAM_BLOCK_S;
          if (s_end > n)
            s_end = n;

          for (r = rb; r < r_end; ++r)
            {
              DATA_TYPE * __restrict B_row = B_[r];
              const DATA_TYPE * __restrict A_row_r = A[r];

              for (s = sb; s < s_end; ++s)
                {
                  const DATA_TYPE * __restrict A_row_s = A[s];
                  DATA_TYPE sum = 0;

                  for (t = 0; t < n; ++t)
                    sum += A_row_r[t] * A_row_s[t];

                  B_row[s] = sum;
                }
            }
        }
    }

  /* Copy B back into A. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE * __restrict A_row = A[r];
      const DATA_TYPE * __restrict B_row = B_[r];

      for (s = 0; s < n; ++s)
	A_row[s] = B_row[s];
    }

  POLYBENCH_FREE_ARRAY(B);
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
   including the call and return. */
static
void kernel_ludcmp[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  /* Restrict-qualified local aliases help the compiler perform
   * better alias analysis and enable more aggressive optimizations.
   * After these are defined, the original parameter names are not
   * used again. */
  DATA_TYPE * __restrict b_ = b;
  DATA_TYPE * __restrict x_ = x;
  DATA_TYPE * __restrict y_ = y;

  (void)n; /* n is accessed through the _PB_N macro. */

#pragma scop
  /* LU factorization using a Doolittle-like algorithm.
   *
   * The structure of the computation is unchanged compared to the
   * original code, but the inner loops are expressed as explicit
   * dot-products (reductions). This improves data locality for row
   * accesses (A[i][k]) and makes it easier for the compiler to
   * vectorize the innermost loops.
   */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE * __restrict A_i = A[i];

    /* Compute the L-part: A[i][j] for j < i. */
    for (j = 0; j < i; j++) {
      DATA_TYPE sum = 0;

      /* Original:
       *   w = A[i][j];
       *   for (k = 0; k < j; k++)
       *     w -= A[i][k] * A[k][j];
       *   A[i][j] = w / A[j][j];
       *
       * We rewrite this as:
       *   sum = sum_{k=0..j-1} A[i][k] * A[k][j];
       *   A[i][j] = (A[i][j] - sum) / A[j][j];
       *
       * The numerical result is identical and the accumulation order
       * over k is unchanged.
       */
#pragma GCC ivdep
      for (k = 0; k < j; k++) {
	sum += A_i[k] * A[k][j];
      }
      A_i[j] = (A_i[j] - sum) / A[j][j];
    }

    /* Compute the U-part: A[i][j] for j >= i (including the diagonal). */
    for (j = i; j < _PB_N; j++) {
      DATA_TYPE sum = 0;

      /* Original:
       *   w = A[i][j];
       *   for (k = 0; k < i; k++)
       *     w -= A[i][k] * A[k][j];
       *   A[i][j] = w;
       *
       * Rewritten as:
       *   sum = sum_{k=0..i-1} A[i][k] * A[k][j];
       *   A[i][j] -= sum;
       */
#pragma GCC ivdep
      for (k = 0; k < i; k++) {
	sum += A_i[k] * A[k][j];
      }
      A_i[j] -= sum;
    }
  }

  /* Forward substitution: solve L * y = b. */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE * __restrict A_i = A[i];
    DATA_TYPE sum = 0;

#pragma GCC ivdep
    for (j = 0; j < i; j++)
      sum += A_i[j] * y_[j];

    y_[i] = b_[i] - sum;
  }

  /* Backward substitution: solve U * x = y. */
  for (i = _PB_N-1; i >= 0; i--) {
    DATA_TYPE * __restrict A_i = A[i];
    DATA_TYPE sum = 0;

#pragma GCC ivdep
    for (j = i+1; j < _PB_N; j++)
      sum += A_i[j] * x_[j];

    x_[i] = (y_[i] - sum) / A_i[i];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}