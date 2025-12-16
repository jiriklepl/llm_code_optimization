/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Optional: enable OpenMP-based parallelization of the initialization. */
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Initialize A as in the original version. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE* Ai = A[i];

      for (j = 0; j <= i; j++)
        Ai[j] = (DATA_TYPE)(-j % n) / n + (DATA_TYPE)1;

      for (j = i+1; j < n; j++)
        Ai[j] = (DATA_TYPE)0;

      Ai[i] = (DATA_TYPE)1;
    }

  /* Make the matrix positive semi-definite.
   *
   * Original code computes:
   *   B[r][s] = sum_t A[r][t] * A[s][t]
   *   A = B
   *
   * We keep exactly the same mathematical operation, but:
   *  - exploit symmetry B[r][s] == B[s][r] to halve the work,
   *  - use row pointers for better cache locality,
   *  - parallelize over rows r with OpenMP (if enabled at compile time),
   *  - avoid the explicit zero-initialization of B by writing each entry once.
   *
   * For every pair (r,s) the accumulation order over t remains
   * strictly increasing from 0 to n-1, as in the original version,
   * so floating-point semantics for each B[r][s] are preserved.
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Compute B = A * A^T.
   * Only the upper triangle (r <= s) is explicitly computed; the
   * lower triangle is filled by symmetry.
   */
#pragma omp parallel for private(s,t) schedule(static)
  for (r = 0; r < n; ++r)
  {
    const DATA_TYPE* Ar = A[r];

    for (s = r; s < n; ++s)
    {
      const DATA_TYPE* As = A[s];
      DATA_TYPE sum = (DATA_TYPE)0;

      /* Dot product of row r and row s of A. */
      for (t = 0; t < n; ++t)
        sum += Ar[t] * As[t];

      (POLYBENCH_ARRAY(B))[r][s] = sum;
      if (s != r)
        (POLYBENCH_ARRAY(B))[s][r] = sum;
    }
  }

  /* Copy B back into A.  Using memcpy works row-wise and keeps
   * the semantics identical to the original double loop copy.
   */
  for (r = 0; r < n; ++r)
    memcpy(A[r],
           (POLYBENCH_ARRAY(B))[r],
           (size_t)n * sizeof(DATA_TYPE));

  POLYBENCH_FREE_ARRAY(B);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky[[gnu::flatten, gnu::noinline]](int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;
  (void)n; /* n is used via the PolyBench macro _PB_N. */

#pragma scop
  /* In-place left-looking Cholesky factorization.
   *
   * Optimizations vs. the baseline:
   *  - Use row pointers (Ai, Aj) to reduce address arithmetic.
   *  - Accumulate into local scalars (sum, diag) to keep values in registers.
   *  - Avoid re-loading A[j][j] and A[i][i] inside inner loops.
   *
   * The loop bounds and the exact sequence of arithmetic operations
   * for each element A[i][j] and A[i][i] are preserved:
   *  - For off-diagonal elements, we still perform
   *        A[i][j] -= A[i][k] * A[j][k]  for k = 0..j-1
   *    in the same order, then divide by A[j][j].
   *  - For diagonal elements, we still perform
   *        A[i][i] -= A[i][k] * A[i][k]  for k = 0..i-1
   *    in the same order, then apply SQRT_FUN.
   */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *Ai = A[i];

    /* Compute L(i, j) for j < i (off-diagonal part). */
    for (j = 0; j < i; j++) {
      DATA_TYPE *Aj = A[j];
      DATA_TYPE sum = Ai[j]; /* start from original A(i,j) */

      for (k = 0; k < j; k++) {
        sum -= Ai[k] * Aj[k];
      }

      Ai[j] = sum / Aj[j];
    }

    /* Compute L(i, i) (diagonal element). */
    DATA_TYPE diag = Ai[i]; /* original A(i,i) */

    for (k = 0; k < i; k++) {
      DATA_TYPE Aik = Ai[k];
      diag -= Aik * Aik;
    }

    Ai[i] = SQRT_FUN(diag);
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}