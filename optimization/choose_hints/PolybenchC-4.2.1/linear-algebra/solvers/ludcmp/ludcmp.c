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

/* Optional tuning parameter: minimal problem size at which we allow
 * OpenMP parallelisation of the U-factor update.  When compiled
 * without OpenMP support, the pragmas are ignored by the compiler.
 */
#ifndef LUDCMP_OMP_MIN_N
# define LUDCMP_OMP_MIN_N 64
#endif


/* Array initialization.
 *
 * This function is not timed by PolyBench, but we still improve its
 * memory access patterns:
 *  - create local restrict-qualified pointers to help alias analysis
 *  - use row pointers when accessing 2D arrays
 * All transformations keep the order of floating-point operations
 * on each scalar element unchanged.
 */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  /* Local "views" with restrict to give the compiler stronger
     aliasing guarantees.  A, b, x, y are allocated independently,
     so they never overlap in memory. */
  DATA_TYPE (*restrict A_mat)[n] = A;
  DATA_TYPE *restrict b_vec      = b;
  DATA_TYPE *restrict x_vec      = x;
  DATA_TYPE *restrict y_vec      = y;

  /* Initialize vectors x, y, b. */
  for (i = 0; i < n; i++)
    {
      x_vec[i] = 0;
      y_vec[i] = 0;
      /* Keep the exact evaluation order as in the original code:
         ((i+1)/fn)/2.0 + 4  (left-associative /). */
      b_vec[i] = (i+1)/fn/2.0 + 4;
    }

  /* Initialize A as a unit-diagonal lower-triangular matrix with an
     extra pattern in the strictly lower part. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *restrict Ai = A_mat[i];
      for (j = 0; j <= i; j++)
        /* We keep the original formula exactly, even though for
           0 <= j < n we have (-j % n) == -j. */
        Ai[j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++)
        Ai[j] = 0;
      Ai[i] = 1;
    }

  /* Make the matrix positive semi-definite.
   * This computes B = A * A^T, then copies B back into A.
   *
   * We keep the outermost loop over t so that, for each B[r][s],
   * the sequence of floating-point additions over t is unchanged
   * compared to the original code (preserving numerical semantics).
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  DATA_TYPE (*restrict B_mat)[n] = POLYBENCH_ARRAY(B);

  /* Zero-initialize B. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE *restrict Br = B_mat[r];
      for (s = 0; s < n; ++s)
        Br[s] = 0;
    }

  /* B[r][s] += A[r][t] * A[s][t] for all r,s,t. */
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      {
        const DATA_TYPE Ar_t = A_mat[r][t];
        DATA_TYPE *restrict Br = B_mat[r];
        for (s = 0; s < n; ++s)
          Br[s] += Ar_t * A_mat[s][t];
      }

  /* Copy B back into A. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE *restrict Ar = A_mat[r];
      DATA_TYPE *restrict Br = B_mat[r];
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
 *  - Introduce restrict-qualified local pointers for A, b, x, y to
 *    help the compiler with alias analysis and enable better
 *    vectorization / register allocation.
 *  - Use row pointers (Ai) to reduce address calculations and improve
 *    cache locality for row-wise accesses.
 *  - Parallelize the computation of the U part (second j-loop) with
 *    OpenMP.  For a fixed i, different columns j >= i are independent,
 *    so they can be computed in parallel without changing the order
 *    of floating-point operations within each element.
 *  - Preserve the exact iteration order of all inner loops, so the
 *    sequence of FP operations for each scalar result remains the same.
 */
static void __attribute__((flatten, noinline))
kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  /* Local restrict-qualified views of the input/output arrays. */
  DATA_TYPE (*restrict a)[n] = A;
  DATA_TYPE *restrict b_vec  = b;
  DATA_TYPE *restrict x_vec  = x;
  DATA_TYPE *restrict y_vec  = y;

#pragma scop
  /* LU factorization in-place (Doolittle, no pivoting). */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *restrict Ai = a[i];

    /* Compute L(i,0..i-1): lower-triangular part of row i.
       This loop is inherently sequential because each Ai[j]
       depends on previously computed Ai[k] for k<j. */
    for (j = 0; j < i; j++) {
      DATA_TYPE w = Ai[j];
      for (k = 0; k < j; k++) {
        /* Use explicit row pointer for A[k] to reduce indexing cost. */
        DATA_TYPE *restrict Ak = a[k];
        w -= Ai[k] * Ak[j];
      }
      Ai[j] = w / a[j][j];
    }

    /* Compute U(i,i..N-1): upper-triangular part of row i.
       For fixed i, each column j >= i is independent, so this
       loop can safely be parallelized if OpenMP is enabled. */
#if defined(_OPENMP)
#   pragma omp parallel for private(k) schedule(static) if (_PB_N >= LUDCMP_OMP_MIN_N)
#endif
    for (j = i; j < _PB_N; j++) {
      DATA_TYPE w = Ai[j];
      for (k = 0; k < i; k++) {
        DATA_TYPE *restrict Ak = a[k];
        w -= Ai[k] * Ak[j];
      }
      Ai[j] = w;
    }
  }

  /* Forward substitution: solve L * y = b. */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *restrict Ai = a[i];
    DATA_TYPE w = b_vec[i];
    for (j = 0; j < i; j++)
      w -= Ai[j] * y_vec[j];
    y_vec[i] = w;
  }

  /* Back substitution: solve U * x = y. */
  for (i = _PB_N-1; i >= 0; i--) {
    DATA_TYPE *restrict Ai = a[i];
    DATA_TYPE w = y_vec[i];
    for (j = i+1; j < _PB_N; j++)
      w -= Ai[j] * x_vec[j];
    x_vec[i] = w / Ai[i];
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