/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stddef.h> /* for size_t */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"

/* Tunable blocking factor for the k dimension (inner product length).
 * By default we process the entire k-dimension in one pass (no real
 * blocking).  You can experiment with cache behavior by compiling with
 * e.g., -DSYRK_K_BLOCK=256.
 */
#ifndef SYRK_K_BLOCK
# define SYRK_K_BLOCK 1024
#endif


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
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

   Optimized version notes
   -----------------------
   The mathematical operation is:

     C := alpha * A * A^T + beta * C,  (lower triangular part only)

   compared to the original implementation, this version:

   1. Fuses the scaling by beta and the update by alpha into a single
      pass over the lower triangle of C, reducing memory traffic.
   2. Computes each C[i][j] as a single dot product over k, keeping
      the accumulator in a register so that C[i][j] is loaded and
      stored only once.
   3. Traverses rows of A contiguously inside the inner k-loop:
        - For fixed (i, j), both A[i][k] and A[j][k] are accessed
          with unit stride, which greatly improves cache and
          prefetch efficiency compared to the original column-wise
          access of A[j][k].
   4. Exposes an optional blocking parameter SYRK_K_BLOCK for the
      k-dimension, which can be tuned for particular cache sizes.
   5. Preserves the original numerical order of operations for each
      C[i][j]: we still do
         C[i][j] = (C[i][j] * beta) + sum_{k=0..m-1}(alpha*A[i][k]*A[j][k])
      with k increasing, but we keep the running sum only in a local
      scalar instead of repeatedly writing back to memory.
*/
static
void kernel_syrk[[gnu::flatten, gnu::noinline]](int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

  /* Local aliases and constants to help the compiler optimize.
     We flatten the 2D arrays into 1D pointers with explicit
     leading dimensions to avoid repeated address arithmetic.
     No 'restrict' is used here so that we do not change the
     aliasing semantics of the original code.
   */
  DATA_TYPE *C_data = &C[0][0];
  DATA_TYPE *A_data = &A[0][0];

  const int n_pb = _PB_N;           /* loop bound for i/j */
  const int m_pb = _PB_M;           /* loop bound for k   */
  const size_t ldC = (size_t)n;     /* leading dimension of C (number of columns) */
  const size_t ldA = (size_t)m;     /* leading dimension of A (number of columns) */

  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

#pragma scop
  /* Row-major traversal of the lower triangular part of C.
     For each (i, j) with j <= i, we compute:
       C[i][j] = beta * C[i][j] + alpha * dot(A[i][:], A[j][:])
  */
  for (i = 0; i < n_pb; i++)
  {
    /* Pointer to the beginning of row i in C and A. */
    DATA_TYPE *Ci = C_data + (size_t)i * ldC;
    DATA_TYPE *Ai = A_data + (size_t)i * ldA;

    for (j = 0; j <= i; j++)
    {
      /* Start from beta * C[i][j] as in the original code. */
      DATA_TYPE cij = Ci[j] * beta_local;

      /* Pointer to the beginning of row j in A. */
      DATA_TYPE *Aj = A_data + (size_t)j * ldA;

      /* Compute the dot product between row i and row j of A,
         scaled by alpha, and accumulate it into cij.

         The inner loop walks k with unit stride over both Ai and Aj,
         improving spatial locality and enabling efficient SIMD
         vectorization.
       */
      int kk;
      for (kk = 0; kk < m_pb; kk += SYRK_K_BLOCK)
      {
        int kend = kk + SYRK_K_BLOCK;
        if (kend > m_pb)
          kend = m_pb;

        /* When SYRK_K_BLOCK >= m_pb (the default), this degenerates
           to a single simple k-loop with no overhead. */
        for (k = kk; k < kend; k++)
        {
          cij += alpha_local * Ai[k] * Aj[k];
        }
      }

      /* Write back the updated value once. */
      Ci[j] = cij;
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

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}