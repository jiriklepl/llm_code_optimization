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

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
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

   Optimizations applied (while preserving original semantics):

   - Use local restrict-qualified pointers to help the compiler
     reason about aliasing and keep rows in registers.
   - Precompute and cache 1 / A[i][i] in an O(N) temporary array
     `inv_diag`, so divisions in the L-update and backward solve
     become cheaper multiplications.
   - Restructure the U-update into a sequence of rank-1 updates:
         for k < i: A[i, i:N) -= L(i,k) * U(k, i:N)
     This changes the loop order from (j, k) to (k, j), giving
     contiguous accesses in the inner j-loop for both A[i,*] and
     A[k,*], improving cache locality and enabling SIMD.
   - Manually unroll short inner loops (k in the L-update and
     j in the U-update) to expose more instruction-level
     parallelism; edge iterations are handled safely.
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

  /* Local aliases with restrict to help the optimizer.
     POLYBENCH_2D(A,...) expands to a VLA type DATA_TYPE A[n][n]. */
  DATA_TYPE (* __restrict__ A_)[n] = A;
  DATA_TYPE * __restrict__ b_ = b;
  DATA_TYPE * __restrict__ x_ = x;
  DATA_TYPE * __restrict__ y_ = y;

  /* Reciprocal of diagonal entries of U.
     Size O(n) << O(n^2), so extra memory is negligible. */
  DATA_TYPE inv_diag[n];

#pragma scop
  /* ---------------------------------------------------------
   * 1. In-place LU factorization without pivoting.
   *    A holds L (strict lower, unit diagonal) and U (upper).
   * --------------------------------------------------------- */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE * __restrict__ Ai = A_[i];

    /* ---- 1.a. Compute L(i, 0..i-1) (strict lower part of row i) ----
       L(i,j) = (A(i,j) - sum_{k<j} L(i,k) * U(k,j)) / U(j,j)
              = (A(i,j) - sum_{k<j} A[i][k] * A[k][j]) * inv_diag[j]
       We keep the original accumulation order over k; only the
       final division is replaced by a multiplication with inv_diag[j]. */
    for (j = 0; j < i; j++) {
      w = Ai[j];

      /* Unrolled reduction over k for better ILP. */
      k = 0;
      for (; k + 3 < j; k += 4) {
        w -= Ai[k]     * A_[k][j];
        w -= Ai[k + 1] * A_[k + 1][j];
        w -= Ai[k + 2] * A_[k + 2][j];
        w -= Ai[k + 3] * A_[k + 3][j];
      }
      for (; k < j; k++) {
        w -= Ai[k] * A_[k][j];
      }

      Ai[j] = w * inv_diag[j];
    }

    /* ---- 1.b. Compute U(i, i..N-1) (upper part of row i) ----
       Original code (per column j):
         w = A[i][j];
         for k < i: w -= A[i][k] * A[k][j];
         A[i][j] = w;

       Here we reorganize it as a sequence of rank-1 updates:
         for k in 0..i-1:
           for j in i..N-1:
             A[i][j] -= A[i][k] * A[k][j];

       For each fixed (i, j), the products are applied in the same
       k-order as in the original code, so floating-point results
       are bit-identical, while the inner j-loop now traverses
       contiguous memory, which is cache- and SIMD-friendly. */
    for (k = 0; k < i; k++) {
      DATA_TYPE Lik = Ai[k];
      DATA_TYPE * __restrict__ Ak = A_[k];

      /* Manual unrolling by 4 over j for better ILP; the remainder
         loop handles the tail when the span is not a multiple of 4. */
      int jj = i;
      for (; jj + 3 < _PB_N; jj += 4) {
        Ai[jj]     -= Lik * Ak[jj];
        Ai[jj + 1] -= Lik * Ak[jj + 1];
        Ai[jj + 2] -= Lik * Ak[jj + 2];
        Ai[jj + 3] -= Lik * Ak[jj + 3];
      }
      for (; jj < _PB_N; jj++) {
        Ai[jj] -= Lik * Ak[jj];
      }
    }

    /* ---- 1.c. Cache reciprocal of diagonal U(i,i) ----
       Used in:
         - later L-rows: L(i2,j) for all i2 > i and j = i
         - backward substitution for x.
       This replaces many divisions by cheaper multiplications. */
    inv_diag[i] = (DATA_TYPE)1.0 / Ai[i];
  }

  /* ---------------------------------------------------------
   * 2. Forward substitution: solve L * y = b
   *    L has unit diagonal; lower part is stored in A.
   * --------------------------------------------------------- */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE * __restrict__ Ai = A_[i];
    w = b_[i];

    for (j = 0; j < i; j++) {
      w -= Ai[j] * y_[j];
    }

    y_[i] = w;
  }

  /* ---------------------------------------------------------
   * 3. Backward substitution: solve U * x = y
   *    U is upper triangular; diagonal reciprocals in inv_diag.
   * --------------------------------------------------------- */
  for (i = _PB_N - 1; i >= 0; i--) {
    DATA_TYPE * __restrict__ Ai = A_[i];
    w = y_[i];

    for (j = i + 1; j < _PB_N; j++) {
      w -= Ai[j] * x_[j];
    }

    x_[i] = w * inv_diag[i];
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