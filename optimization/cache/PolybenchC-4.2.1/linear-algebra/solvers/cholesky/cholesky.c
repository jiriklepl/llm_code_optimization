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
#include <stddef.h> /* for size_t */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"

/* Tunable block size for the dot-product dimension (t) in the
 * positive semi-definite initialization kernel A <- A * A^T.
 * This only affects performance, not the numerical result.
 */
#ifndef INIT_T_BLOCK
# define INIT_T_BLOCK 64
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Treat the 2-D array as a single contiguous row-major buffer.
     Using a restrict-qualified 1-D pointer improves alias analysis
     and lets us manage the indexing explicitly. */
  DATA_TYPE* restrict A_ = (DATA_TYPE*)A;

  /* Precompute 1/n once; used in the lower-triangular initialization. */
  const DATA_TYPE inv_n = (DATA_TYPE)1.0 / (DATA_TYPE)n;

  /* Initial lower-triangular matrix setup. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE* restrict Ai = A_ + (size_t)i * (size_t)n;

      /* Lower-triangular part (including the diagonal).
         Original expression:
             A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
         For 0 <= j < n, (-j % n) == -j in C, hence this simplifies to:
             A[i][j] = 1 - (DATA_TYPE)j / n;
         We use that simpler and cheaper form here. */
      for (j = 0; j <= i; j++)
        Ai[j] = (DATA_TYPE)1.0 - ((DATA_TYPE)j) * inv_n;

      /* Strictly upper-triangular part. */
      for (j = i + 1; j < n; j++)
        Ai[j] = (DATA_TYPE)0.0;

      /* Ensure diagonal is exactly 1. */
      Ai[i] = (DATA_TYPE)1.0;
    }

  /* Make the matrix positive semi-definite: A <- A * A^T.
     We compute B = A * A^T as:
       B[r][s] = sum_{t=0}^{n-1} A[r][t] * A[s][t]
     then copy B back into A.

     Optimizations compared to the original version:
       - Use explicit row pointers into 1-D buffers for A and B
         to minimize index arithmetic inside inner loops.
       - Exploit symmetry: compute only for s >= r and mirror to (s, r),
         halving the number of dot products.
       - Use a tunable blocking factor along the t-dimension (INIT_T_BLOCK)
         for better cache locality on long rows.
       - Use an explicit accumulator 'sum' to keep the innermost loop
         free of loop-carried dependencies through memory, improving
         vectorization friendliness.

     The accumulation order over t for each (r,s) element remains
     strictly increasing from 0 to n-1, so the numerical semantics
     match the original code. */

  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  DATA_TYPE* restrict B_ = (DATA_TYPE*)POLYBENCH_ARRAY(B);

  int r, s, t;

  for (r = 0; r < n; ++r)
    {
      DATA_TYPE* restrict Br = B_ + (size_t)r * (size_t)n;
      DATA_TYPE const* restrict Ar = A_ + (size_t)r * (size_t)n;

      /* Exploit symmetry: compute for s >= r and mirror to s < r. */
      for (s = r; s < n; ++s)
        {
          DATA_TYPE* restrict Bs = B_ + (size_t)s * (size_t)n;
          DATA_TYPE const* restrict As = A_ + (size_t)s * (size_t)n;

          DATA_TYPE sum = (DATA_TYPE)0.0;

          /* Block the dot-product dimension (t) to improve cache
             behavior when n is large. The effective order in t
             is still 0,1,2,...,n-1, so we preserve the original
             reduction order for each (r,s). */
          for (int tt = 0; tt < n; tt += INIT_T_BLOCK)
            {
              int t_end = tt + INIT_T_BLOCK;
              if (t_end > n)
                t_end = n;

              for (t = tt; t < t_end; ++t)
                sum += Ar[t] * As[t];
            }

          /* Write the symmetric pair. For r == s these refer to the
             same location, so the second store is a no-op. */
          Br[s] = sum;
          Bs[r] = sum;
        }
    }

  /* Copy B back into A. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE* restrict Ar = A_ + (size_t)r * (size_t)n;
      DATA_TYPE const* restrict Br = B_ + (size_t)r * (size_t)n;

      for (s = 0; s < n; ++s)
        Ar[s] = Br[s];
    }

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

  /* As in init_array, view the 2-D array as a contiguous 1-D buffer
     to reduce index arithmetic. */
  DATA_TYPE* restrict A_ = (DATA_TYPE*)A;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    /* Pointer to row i. */
    DATA_TYPE* restrict Ai = A_ + (size_t)i * (size_t)n;

    /* Off-diagonal elements in row i (j < i). */
    for (j = 0; j < i; j++) {
      DATA_TYPE* restrict Aj = A_ + (size_t)j * (size_t)n;

      /* Original code:
           for (k = 0; k < j; k++)
             A[i][j] -= A[i][k] * A[j][k];
           A[i][j] /= A[j][j];

         We rewrite the update as a scalar reduction:
           dot = A[i][j];
           for k: dot -= A[i][k] * A[j][k];
           A[i][j] = dot / A[j][j];

         This removes the loop-carried dependence through A[i][j],
         so the compiler can more easily vectorize the inner loop
         as a pure reduction on 'dot'. The arithmetic order over k
         (0 .. j-1) is preserved exactly. */
      DATA_TYPE dot = Ai[j];

#pragma GCC ivdep
      for (k = 0; k < j; k++) {
        dot -= Ai[k] * Aj[k];
      }

      Ai[j] = dot / Aj[j];
    }

    /* Diagonal element A[i][i]. */
    DATA_TYPE diag = Ai[i];

    /* Original:
         for (k = 0; k < i; k++)
           A[i][i] -= A[i][k] * A[i][k];
       Same reduction-style rewrite. */
#pragma GCC ivdep
    for (k = 0; k < i; k++) {
      diag -= Ai[k] * Ai[k];
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