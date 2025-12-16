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

/* --------------------------------------------------------------------
 * Tunable parameters for this optimized version.
 *
 * LUDCMP_OMP_MIN_INIT_N:
 *   Minimal matrix size at which OpenMP is enabled for the
 *   positive-semidefinite (SPD) construction in init_array().
 *   For smaller n, the overhead of starting threads can dominate.
 * ------------------------------------------------------------------*/
#ifndef LUDCMP_OMP_MIN_INIT_N
# define LUDCMP_OMP_MIN_INIT_N 64
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

  /* Initialize vectors. Independent across i. */
  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  /* Initialize A to a lower-triangular matrix with ones on the diagonal. */
  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++)
	A[i][j] = 0;

      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite.
   *
   * Original code computes:
   *   B[r][s] = sum_t A[r][t] * A[s][t]
   *
   * which is B = A * A^T. Here we:
   *   - Exploit the symmetry B[r][s] == B[s][r] to halve the work.
   *   - Use a more cache-friendly order (outer loops over rows,
   *     inner loop is a dot product over contiguous elements).
   *   - Optionally parallelize over rows with OpenMP.
   *
   * Semantics are preserved; only the summation order changes slightly,
   * which may lead to minimal floating-point round-off differences.
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Create local restrict-qualified aliases to help the compiler. */
  DATA_TYPE (*restrict A2)[n] = A;
  DATA_TYPE (*restrict B2)[n] = POLYBENCH_ARRAY(B);

  /* Compute B = A * A^T using only the upper triangle and mirror it. */
#ifdef _OPENMP
# pragma omp parallel for schedule(static) if (n >= LUDCMP_OMP_MIN_INIT_N) private(s,t)
#endif
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE *restrict Br = B2[r];
      const DATA_TYPE *restrict Ar = A2[r];

      for (s = r; s < n; ++s)
	{
	  const DATA_TYPE *restrict As = A2[s];
	  DATA_TYPE sum = (DATA_TYPE)0;

	  /* Dot product of row r and row s. */
#pragma GCC ivdep
	  for (t = 0; t < n; ++t)
	    sum += Ar[t] * As[t];

	  Br[s] = sum;

	  /* Use symmetry: B[s][r] = B[r][s] for r != s. */
	  if (s != r)
	    B2[s][r] = sum;
	}
    }

  /* Copy B back into A. This preserves the original semantics A := B.  */
#ifdef _OPENMP
# pragma omp parallel for schedule(static) if (n >= LUDCMP_OMP_MIN_INIT_N) private(s)
#endif
  for (r = 0; r < n; ++r)
    {
      const DATA_TYPE *restrict Br = B2[r];
      DATA_TYPE *restrict Ar = A2[r];

#pragma GCC ivdep
      for (s = 0; s < n; ++s)
	Ar[s] = Br[s];
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
   including the call and return.
 *
 * Optimizations applied:
 *  - Introduce restrict-qualified local aliases for all arrays
 *    to improve alias analysis.
 *  - Reduce repeated address calculations by caching row pointers.
 *  - Reorder the update of the upper-triangular part (U) into a
 *    sequence of rank-1 updates:
 *
 *        for k < i:  A[i, i..] -= L[i,k] * U[k, i..]
 *
 *    This improves data locality and allows the compiler to
 *    better vectorize the inner loop over j.
 *  - Add vectorization hints (GCC ivdep) for inner dot-product
 *    style loops.
 */
static
void kernel_ludcmp[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;
  DATA_TYPE w;

  /* Local restrict-qualified aliases. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE *restrict b_ = b;
  DATA_TYPE *restrict x_ = x;
  DATA_TYPE *restrict y_ = y;

#pragma scop
  /* LU factorization without pivoting: A = L * U in-place. */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *restrict A_i = A_[i];

    /* Compute the strictly lower part L(i,0..i-1). */
    for (j = 0; j < i; j++) {
      const DATA_TYPE *restrict A_j = A_[j];

      w = A_i[j];

      /* w -= dot( A_i[0..j-1], A_j[0..j-1] ) */
#pragma GCC ivdep
      for (k = 0; k < j; k++) {
        w -= A_i[k] * A_j[k];
      }

      A_i[j] = w / A_j[j];
    }

    /* Compute the upper part U(i,i..N-1).
     *
     * Original code:
     *   for (j = i; j < N; ++j) {
     *     w = A[i][j];
     *     for (k = 0; k < i; ++k)
     *       w -= A[i][k] * A[k][j];
     *     A[i][j] = w;
     *   }
     *
     * Reordered to a rank-1 update formulation:
     *   for (k = 0; k < i; ++k) {
     *     Lik = A[i][k];
     *     for (j = i; j < N; ++j)
     *       A[i][j] -= Lik * A[k][j];
     *   }
     *
     * Algebraically identical (only loop order changes), but
     * improves cache reuse of A[k][j] and A[i][j].
     */
    for (k = 0; k < i; k++) {
      const DATA_TYPE Lik = A_i[k];
      const DATA_TYPE *restrict A_k = A_[k];

#pragma GCC ivdep
      for (j = i; j < _PB_N; j++) {
        A_i[j] -= Lik * A_k[j];
      }
    }
  }

  /* Forward substitution: solve L * y = b. */
  for (i = 0; i < _PB_N; i++) {
    const DATA_TYPE *restrict A_i = A_[i];
    w = b_[i];

#pragma GCC ivdep
    for (j = 0; j < i; j++)
      w -= A_i[j] * y_[j];

    y_[i] = w;
  }

  /* Backward substitution: solve U * x = y. */
  for (i = _PB_N-1; i >= 0; i--) {
    const DATA_TYPE *restrict A_i = A_[i];
    w = y_[i];

#pragma GCC ivdep
    for (j = i+1; j < _PB_N; j++)
      w -= A_i[j] * x_[j];

    x_[i] = w / A_i[i];
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