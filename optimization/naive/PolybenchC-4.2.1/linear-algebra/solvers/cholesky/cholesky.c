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

  /* Restrict-qualified alias to help the compiler optimize accesses. */
  DATA_TYPE (*restrict A_)[n] = A;

  /* Initial (non-SPD) matrix. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *restrict Ai = A_[i];

      for (j = 0; j <= i; j++)
        Ai[j] = (DATA_TYPE)(-j % n) / n + 1;

      for (j = i+1; j < n; j++) {
        Ai[j] = (DATA_TYPE)0;
      }

      Ai[i] = (DATA_TYPE)1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  DATA_TYPE (*restrict B_)[n] = POLYBENCH_ARRAY(B);

  /* Zero-initialize B in a single contiguous operation. */
  memset(&B_[0][0], 0, (size_t)n * (size_t)n * sizeof(DATA_TYPE));

  /* Compute B = A * A^T.
     Loop order (t, r, s) is preserved so that, for each (r,s),
     the accumulation over t happens in the same order as in
     the original code. We only introduce local pointers and
     a scalar temporary to reduce redundant loads. */
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      {
        DATA_TYPE A_rt = A_[r][t];
        DATA_TYPE *restrict B_r = B_[r];

        for (s = 0; s < n; ++s)
          B_r[s] += A_rt * A_[s][t];
      }

  /* Copy B back to A, row by row (cache-friendly). */
  for (r = 0; r < n; ++r)
    memcpy(A_[r], B_[r], (size_t)n * sizeof(DATA_TYPE));

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
   including the call and return.

   Optimizations applied:

   - Use a restrict-qualified alias for A to aid the compiler.
   - Introduce row pointers (Ai, Aj) to improve cache locality and
     reduce address arithmetic.
   - Fuse the diagonal update into the main j-loop: instead of a
     separate loop
         for (k = 0; k < i; k++) A[i][i] -= A[i][k] * A[i][k];
     we update the diagonal incrementally as soon as A[i][j] is
     computed. This keeps *exactly* the same per-element update
     order for the diagonal (increasing j), but avoids an extra
     pass over the row.
   - Manually unroll the inner dot product by a factor of 4 while
     preserving the exact original order of floating-point
     operations on the accumulation variable.
*/
static
void kernel_cholesky[[gnu::flatten, gnu::noinline]](int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Alias with restrict to communicate non-aliasing to the compiler. */
  DATA_TYPE (*restrict A_)[n] = A;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
     DATA_TYPE *restrict Ai = A_[i];

     /* Start from the original diagonal element; it will be updated
        incrementally as we compute the off-diagonal entries in this row. */
     DATA_TYPE diag = Ai[i];

     /* Off-diagonal part (column 0 .. i-1 of row i). */
     for (j = 0; j < i; j++) {
        DATA_TYPE *restrict Aj = A_[j];

        /* Begin with the current A[i][j] value. */
        DATA_TYPE sum = Ai[j];

        /* Compute: sum -= A[i][k] * A[j][k] for k = 0 .. j-1
           This is a dot product between the first j elements of rows i and j.
           The operations on 'sum' occur in the same order as the original
           code (increasing k); the loop is simply manually unrolled to
           expose more instruction-level parallelism. */
        int k = 0;
        int k_unroll = j & ~3; /* largest multiple of 4 <= j */

        for (; k < k_unroll; k += 4) {
           sum -= Ai[k]     * Aj[k];
           sum -= Ai[k + 1] * Aj[k + 1];
           sum -= Ai[k + 2] * Aj[k + 2];
           sum -= Ai[k + 3] * Aj[k + 3];
        }
        for (; k < j; ++k) {
           sum -= Ai[k] * Aj[k];
        }

        /* Normalize by the already computed diagonal A[j][j]. */
        sum /= Aj[j];
        Ai[j] = sum;

        /* Incremental diagonal update:
             diag_new = diag_old - (A[i][j])^2
           Executed once per j in ascending order, equivalent to the
           original separate loop:
             diag = A[i][i];
             for (k = 0; k < i; ++k) diag -= A[i][k] * A[i][k];
        */
        diag -= sum * sum;
     }

     /* Finalize diagonal element. */
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