/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"

/* -------------------------------------------------------------------------
 * Tunable kernel parameters.
 *
 * K_TILE controls blocking of the reduction over k in the main kernel.
 * It can be overridden at compile time, e.g.:
 *
 *   gcc -O3 -DK_TILE=128 ...
 *
 * so that users can tune for a particular machine/cache.
 * ------------------------------------------------------------------------- */
#ifndef K_TILE
# define K_TILE 64
#endif


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  /* Initialize A and B: row-major (j innermost) for good locality. */
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i * j + 1) % n) / n;
      B[i][j] = (DATA_TYPE) ((i * j + 2) % m) / m;
    }

  /* Initialize C: again row-major. */
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i * j + 3) % n) / m;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Original operation (symmetric rank‑2k update, lower triangle):
     C(i,j) = beta * C(i,j)
              + sum_k [ alpha * A(j,k) * B(i,k) + alpha * B(j,k) * A(i,k) ]
   for 0 <= j <= i < N.

   Optimizations:
   - Reorder loops from (i, j; then i, k, j) to (i, j, k):
       * Keeps k as the innermost loop so that A(*,k) and B(*,k)
         are accessed with the last index varying fastest
         (contiguous in row-major layout).
       * For each (i,j), C(i,j) is loaded once, scaled by beta,
         accumulated in a register across all k, and stored once.
         This removes repeated loads/stores of C inside the k-loop.
       * For each (i,j), the sequence over k is still 0..M-1, so the
         reduction order over k for a fixed (i,j) is unchanged.
   - Use local restrict-qualified pointers to help the compiler with
     alias analysis while preserving the original behavior (A, B, C
     are distinct arrays in PolyBench).
   - Simple blocking over k (K_TILE) to improve cache reuse for large M.

   The arithmetic expression inside the innermost loop is kept in the
   same parenthesization as the original code to preserve the exact
   floating-point evaluation order of the multiplications.
*/
static
void kernel_syr2k [[gnu::flatten, gnu::noinline]] (int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j, k;

  /* Local aliases with explicit restrict to aid optimization.
     Types match the VLA parameter types (n x n, n x m). */
  DATA_TYPE (*restrict C_)[n] = (DATA_TYPE (*restrict)[n]) C;
  DATA_TYPE (*restrict A_)[m] = (DATA_TYPE (*restrict)[m]) A;
  DATA_TYPE (*restrict B_)[m] = (DATA_TYPE (*restrict)[m]) B;

  /* PolyBench may set _PB_N and _PB_M slightly smaller than n,m.
     We keep loop bounds based on these macros, as in the original. */
  const int n_it = _PB_N;
  const int m_it = _PB_M;

#pragma scop
  for (i = 0; i < n_it; i++) {

    /* Row pointers for i, reused across all j in this row of C. */
    DATA_TYPE *Ai = A_[i];
    DATA_TYPE *Bi = B_[i];

    /* Only the lower-triangular part (j <= i) is updated. */
    for (j = 0; j <= i; j++) {

      DATA_TYPE *Aj = A_[j];
      DATA_TYPE *Bj = B_[j];

      /* Load C(i,j) once, apply beta, and hold in a register. */
      DATA_TYPE cij = C_[i][j] * beta;

      /* Reduction over k, blocked by K_TILE for cache locality. */
      for (int kk = 0; kk < m_it; kk += K_TILE) {
        const int kend = (kk + K_TILE < m_it) ? (kk + K_TILE) : m_it;

        /* k-loop is innermost: contiguous access to A and B rows. */
        for (k = kk; k < kend; k++) {
          /* Expression is written to preserve original FP grouping:
               A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k]
             which is parsed as:
               ((A[j][k] * alpha) * B[i][k]) +
               ((B[j][k] * alpha) * A[i][k])
          */
          cij += Aj[k] * alpha * Bi[k]
               + Bj[k] * alpha * Ai[k];
        }
      }

      /* Store the fully accumulated value back once. */
      C_[i][j] = cij;
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syr2k (n, m,
		alpha, beta,
		POLYBENCH_ARRAY(C),
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}