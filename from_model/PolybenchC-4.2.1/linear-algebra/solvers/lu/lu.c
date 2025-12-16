/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"


/* Array initialization.
 *
 * 1) Build a simple lower-triangular matrix A.
 * 2) Replace it by a symmetric positive semi-definite matrix A := A * A^T.
 *
 * Step 2 is implemented as a cache-friendly, symmetry-exploiting Gram
 * product. The mathematical result is identical to the original code:
 *   B[r][s] = sum_t A[r][t] * A[s][t]
 *   A       = B
 */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Original lower-triangular initialization (unchanged). */
  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite.
     We compute B = A * A^T as a Gram matrix, then copy B back into A.
     This version:
       - exploits symmetry (computes only half of B and mirrors it),
       - uses contiguous accesses in the inner loop,
       - keeps the per-element accumulation order over t identical to
         the original code (t = 0..n-1), so each B[r][s] is bitwise
         identical to the original implementation. */
  {
    int r, s, t;
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

    DATA_TYPE (* restrict a)[n]  = A;
    DATA_TYPE (* restrict b)[n]  = POLYBENCH_ARRAY(B);

    /* Compute upper triangle (including diagonal) of B, then mirror. */
    for (r = 0; r < n; ++r)
      {
        DATA_TYPE * restrict Ar = a[r];
        for (s = r; s < n; ++s)
          {
            DATA_TYPE sum = (DATA_TYPE)0;
            DATA_TYPE * restrict As = a[s];

            /* Sum over t in strictly increasing order, as in the
               original triple-nested loop (t outermost there). */
            for (t = 0; t < n; ++t)
              sum += Ar[t] * As[t];

            b[r][s] = sum;
            b[s][r] = sum;
          }
      }

    /* Copy B back to A (row-wise, cache friendly). */
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
        a[r][s] = b[r][s];

    POLYBENCH_FREE_ARRAY(B);
  }
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
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   This implements an in-place LU factorization without pivoting:

     - Strict lower triangle (i > j): L(i,j), with unit diagonal implied.
     - Upper triangle and diagonal (i <= j): U(i,j).

   Compared to the original kernel:

     * The lower-part computation (L) keeps the same algorithm and
       dependence structure, but uses an explicit accumulator and
       small unrolling over k for better register reuse and ILP.

     * The upper-part computation (U) is transformed from

         for (j = i; j < N; ++j)
           for (k = 0; k < i; ++k)
             A[i][j] -= A[i][k] * A[k][j];

       into a rank-1 style update with k outermost and j innermost,
       plus tiling and unrolling along j:

         for (k = 0; k < i; ++k) {
           lik = A[i][k];
           for (j tiles starting at i)
             for (j in tile)
               A[i][j] -= lik * A[k][j];
         }

       For a fixed (i, j), the sequence of k-values is still 0..i-1 in
       the same order as in the original code. Therefore, each element
       A[i][j] sees exactly the same sequence of floating-point
       operations; only their interleaving across different columns j
       changes. This preserves the original computation semantics while
       greatly improving data locality and vectorization opportunities. */
static
void kernel_lu[[gnu::flatten, gnu::noinline]](int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

  /* Alias A through a restrict-qualified pointer to help the compiler
     reason about aliasing and enable more aggressive optimizations. */
  DATA_TYPE (* restrict a)[n] = A;

  /* Column tile size for the upper update.
     64 is a reasonable default for modern x86-64 caches, but this is a
     tuning parameter and does not affect correctness. */
  const int BJ = 64;

#pragma scop
  for (i = 0; i < _PB_N; i++) {

    /* ---- Strict lower triangle: compute L(i, j) for j < i ----
       Original code:

         for (j = 0; j < i; j++) {
           for (k = 0; k < j; k++)
             A[i][j] -= A[i][k] * A[k][j];
           A[i][j] /= A[j][j];
         }

       We keep the same left-looking structure and evaluation order of
       the inner products, but use an explicit accumulator and unroll
       the k-loop by 4 to improve register reuse and ILP. */
    for (j = 0; j < i; j++) {
      DATA_TYPE sum = a[i][j];

      int k_end    = j;
      int k_unroll = (k_end / 4) * 4;

      for (k = 0; k < k_unroll; k += 4) {
        sum -= a[i][k    ] * a[k    ][j];
        sum -= a[i][k + 1] * a[k + 1][j];
        sum -= a[i][k + 2] * a[k + 2][j];
        sum -= a[i][k + 3] * a[k + 3][j];
      }
      for (; k < k_end; ++k) {
        sum -= a[i][k] * a[k][j];
      }

      sum /= a[j][j];
      a[i][j] = sum;
    }

    /* ---- Upper triangle (including diagonal): compute U(i, j) for j >= i ----
       Original code:

         for (j = i; j < _PB_N; j++) {
           for (k = 0; k < i; k++)
             A[i][j] -= A[i][k] * A[k][j];
         }

       We legally interchange the j and k loops (and then tile j) to
       obtain a cache-friendly saxpy-like update:

         for (k = 0; k < i; k++) {
           lik = A[i][k];
           for (j = i; j < _PB_N; j++)
             A[i][j] -= lik * A[k][j];
         }

       Because k is still traversed in increasing order for each fixed
       (i, j), the sequence of updates to each A[i][j] is identical to
       the original, preserving floating-point evaluation order. */
    for (k = 0; k < i; ++k) {
      DATA_TYPE lik = a[i][k];  /* L(i, k) is already computed above */

      /* Process row segments [jb, j_end) in tiles to improve cache
         behavior and make the inner loop large enough for efficient
         SIMD/vectorization. */
      int jb;
      for (jb = i; jb < _PB_N; jb += BJ) {
        int j_end = jb + BJ;
        if (j_end > _PB_N)
          j_end = _PB_N;

        int jj;

        /* Unroll the inner j-loop by 4. Accesses to both a[i][*] and
           a[k][*] are contiguous along j, which is ideal for the
           hardware prefetcher and SIMD units. */
        for (jj = jb; jj + 3 < j_end; jj += 4) {
          a[i][jj    ] -= lik * a[k][jj    ];
          a[i][jj + 1] -= lik * a[k][jj + 1];
          a[i][jj + 2] -= lik * a[k][jj + 2];
          a[i][jj + 3] -= lik * a[k][jj + 3];
        }
        for (; jj < j_end; ++jj) {
          a[i][jj] -= lik * a[k][jj];
        }
      }
    }
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
  kernel_lu (n, POLYBENCH_ARRAY(A));

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