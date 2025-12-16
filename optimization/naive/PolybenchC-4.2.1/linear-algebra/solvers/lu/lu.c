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


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Initialize A to a lower-triangular matrix with 1s on the diagonal. */
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
     Not necessary for LU, but using same code as cholesky. */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Compute B = A * A^T explicitly using a dot-product form.
     Compared to the original outer-product style:
         for t
           for r
             for s
               B[r][s] += A[r][t] * A[s][t];
     this version:
       - touches each B[r][s] only once (better cache behavior for B),
       - still accumulates over t in ascending order for every (r,s),
         so the sequence of floating‑point operations for each element
         is identical and numerical results are preserved. */
  for (r = 0; r < n; ++r)
    {
      for (s = 0; s < n; ++s)
        {
          DATA_TYPE sum = (DATA_TYPE)0;
          for (t = 0; t < n; ++t)
            sum += A[r][t] * A[s][t];
          (POLYBENCH_ARRAY(B))[r][s] = sum;
        }
    }

  /* Copy back into A. */
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      A[r][s] = (POLYBENCH_ARRAY(B))[r][s];

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

   This kernel performs an in-place LU decomposition without pivoting:
   for each row i, it first computes the L entries A[i][j] for j < i and
   then the U entries A[i][j] for j >= i.

   The implementation below improves data locality in the U-update phase
   by interchanging the j and k loops (k outer, j inner). For every
   element A[i][j], the sequence of updates over k is still in strictly
   increasing order, so the floating-point operation order for each
   element is unchanged relative to the original code. */
static
void __attribute__((noinline))
kernel_lu(int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* Cache the current row pointer once per i to reduce address
         arithmetic in inner loops. */
      DATA_TYPE *Ai = A[i];

      /* ----- Compute L part: columns j < i ----- */
      for (j = 0; j < i; j++)
        {
          /* Original computation:
               for (k = 0; k < j; k++)
                 A[i][j] -= A[i][k] * A[k][j];
               A[i][j] /= A[j][j];

             We keep the same k-iteration order and accumulate into a
             local variable to give the compiler more optimization
             freedom and reduce repeated loads/stores of A[i][j]. */
          DATA_TYPE sum = Ai[j];
          for (k = 0; k < j; k++)
            sum -= Ai[k] * A[k][j];
          Ai[j] = sum / A[j][j];
        }

      /* ----- Compute U part: columns j >= i ----- */
      /* Original code (j outer, k inner):
             for (j = i; j < _PB_N; j++)
               for (k = 0; k < i; k++)
                 A[i][j] -= A[i][k] * A[k][j];

         We interchange the j and k loops to improve reuse of:
           - L entry A[i][k], and
           - row A[k][*] of U.
         For a fixed (i,j), the updates are still applied with k
         increasing from 0 to i-1, so each A[i][j] sees exactly the same
         arithmetic sequence as in the original code. */
      for (k = 0; k < i; k++)
        {
          const DATA_TYPE aik = Ai[k];   /* L[i][k] */
          const DATA_TYPE *Ak = A[k];    /* row k of U */

          for (j = i; j < _PB_N; j++)
            Ai[j] -= aik * Ak[j];
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