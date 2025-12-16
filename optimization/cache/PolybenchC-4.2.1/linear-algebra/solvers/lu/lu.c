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

/* -------------------------------------------------------------------------
 * Tunable blocking parameter for the LU kernel.
 *
 * LU_BLOCK_J:
 *   - Block size (in columns) used when updating the U-factor part
 *     of the matrix in kernel_lu.
 *   - Can be overridden at compile time, e.g.:
 *
 *       gcc -O3 -DNDEBUG -DLU_BLOCK_J=128 ...
 *
 *   - If set to 0, the blocked variant is disabled and an unblocked
 *     update is used instead.
 * ------------------------------------------------------------------------- */
#ifndef LU_BLOCK_J
# define LU_BLOCK_J 64
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Use a local restrict-qualified pointer to help the compiler
     with alias analysis and improve optimization. */
  DATA_TYPE (*restrict A_)[n] = A;

  /* Original lower-triangular initialization of A. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *restrict Ai = A_[i];

      for (j = 0; j <= i; j++)
	Ai[j] = (DATA_TYPE)(-j % n) / n + 1;

      for (j = i+1; j < n; j++)
	Ai[j] = 0;

      Ai[i] = 1;
    }

  /* Make the matrix positive semi-definite.
     (Same computation as original: B = A * A^T; A <- B.) */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  DATA_TYPE (*restrict B_)[n] = POLYBENCH_ARRAY(B);

  /* Initialize B to zero (same as original). */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE *restrict Br = B_[r];
      for (s = 0; s < n; ++s)
        Br[s] = 0;
    }

  /* Rank-1 updates: B[r][s] += A[r][t] * A[s][t].
     Loop order is kept identical (t, r, s) so that, for each
     element B[r][s], the accumulation over t happens in the same
     order as in the original code (preserving FP semantics). */
  for (t = 0; t < n; ++t)
    {
      for (r = 0; r < n; ++r)
        {
          const DATA_TYPE Art = A_[r][t];
          DATA_TYPE *restrict Br = B_[r];
          for (s = 0; s < n; ++s)
            Br[s] += Art * A_[s][t];
        }
    }

  /* Copy B back into A. Element-wise copy is replaced by memcpy
     per row, which is semantically equivalent because A and B
     are disjoint arrays. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE *restrict Ar = A_[r];
      const DATA_TYPE *restrict Br = B_[r];
      memcpy(Ar, Br, (size_t)n * sizeof(DATA_TYPE));
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
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Introduce a local restrict-qualified pointer to the matrix
     to improve alias analysis.
   - For the U-part update (j >= i), we reorder loops from (j, k)
     to (k, j) so that:
         * For each fixed column j, the sequence of operations
           over k is still k = 0..i-1 in the same order as in
           the original code. This preserves floating-point
           semantics per element A[i][j].
         * Accesses become contiguous in the inner j-loop for both
           row i and row k, improving cache locality.
   - Optional blocking over the j-dimension controlled by LU_BLOCK_J.
*/
static
void kernel_lu[[gnu::flatten, gnu::noinline]](int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  /* Local alias with restrict to improve the compiler's ability to
     optimize memory accesses. */
  DATA_TYPE (*restrict A_)[n] = A;

#pragma scop
  const int nPB = _PB_N;

  for (int i = 0; i < nPB; i++) {

    DATA_TYPE *restrict Ai = A_[i];

    /* Compute L part: columns 0..i-1 of row i. */
    for (int j = 0; j < i; j++) {

      /* Start from the current element, then subtract the inner
         product over k. Using a scalar accumulator avoids repeated
         loads/stores of Ai[j] inside the k-loop. */
      DATA_TYPE sum = Ai[j];

      /* Inner loop over k left in the original order (0..j-1) to
         preserve the exact sequence of floating-point operations. */
      for (int k = 0; k < j; k++) {
        sum -= Ai[k] * A_[k][j];
      }

      /* Divide by the diagonal element of U (A[j][j]) as in the
         original algorithm. */
      sum /= A_[j][j];
      Ai[j] = sum;
    }

    /* Compute U part: columns i..nPB-1 of row i.

       We reorder the loops from:

           for (j = i; j < nPB; j++)
             for (k = 0; k < i; k++)
               Ai[j] -= Ai[k] * A_[k][j];

       to:

           for (k = 0; k < i; k++)
             for (j = i; j < nPB; j++)
               Ai[j] -= Ai[k] * A_[k][j];

       or a blocked variant thereof.

       For each fixed j, the sequence over k is still k = 0..i-1 in
       increasing order, so A[i][j] sees the same arithmetic
       operations in the same order; we only interleave different
       columns, which does not change per-element semantics. */

#if LU_BLOCK_J > 0
    /* Blocked variant in the j-dimension for better cache use. */
    const int jb = LU_BLOCK_J;

    for (int jj = i; jj < nPB; jj += jb) {
      const int j_end = (jj + jb < nPB) ? (jj + jb) : nPB;

      for (int k = 0; k < i; k++) {
        const DATA_TYPE Lik = Ai[k];              /* L(i,k) */
        const DATA_TYPE *restrict Ak = A_[k];     /* row k */

        for (int j = jj; j < j_end; j++) {
          Ai[j] -= Lik * Ak[j];
        }
      }
    }
#else
    /* Simple unblocked variant. */
    for (int k = 0; k < i; k++) {
      const DATA_TYPE Lik = Ai[k];              /* L(i,k) */
      const DATA_TYPE *restrict Ak = A_[k];     /* row k */

      for (int j = i; j < nPB; j++) {
        Ai[j] -= Lik * Ak[j];
      }
    }
#endif
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