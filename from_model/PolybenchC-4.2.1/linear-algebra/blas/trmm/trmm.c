/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trmm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

  /*
   * Optimized implementation of:
   *
   *   SIDE   = 'L'
   *   UPLO   = 'L'
   *   TRANSA = 'T'
   *   DIAG   = 'U'
   *
   * => B := alpha * A^T * B
   *
   * Original kernel (row-major storage):
   *
   *   for i in 0..M-1
   *     for j in 0..N-1
   *       for k in i+1..M-1
   *         B[i][j] += A[k][i] * B[k][j];
   *       B[i][j] *= alpha;
   *
   * This code keeps the same mathematical result but restructures the
   * computation for better data locality and vectorization:
   *
   *   1) Rewrite the triangular update in (k, i, j) order:
   *        for k in 1..M-1
   *          for i in 0..k-1
   *            for j in 0..N-1
   *              B[i][j] += A[k][i] * B[k][j];
   *
   *      This is algebraically identical: each pair (i,k) with k>i is
   *      visited exactly once. For a fixed (i,j), the contributions
   *      from k are still applied in increasing k.
   *
   *   2) Hoist the scaling by alpha:
   *        B_out = alpha * ( B_in + sum_{k>i} A[k,i] * B_in[k,*] )
   *      We first compute the sum (the "ACCUM" phase) and then apply
   *      a separate pass to scale B by alpha (the "SCALE" phase).
   *      Scaling occurs only after the last use of each B[k][j] as a
   *      source term, so the result is identical to the original code.
   *
   *   3) Use blocking and a SAXPY-like inner kernel:
   *        - Inner-most loop is over j (columns of B) so that B rows
   *          are accessed with unit stride (row-major layout).
   *        - A[k][i] is loaded once per (k,i) and reused over j.
   *        - We tile k and j so that the working rows of B and panels
   *          of A fit better into cache.
   *
   *   4) Use linearized pointers with 'restrict' to help the compiler
   *      generate efficient vectorized code.
   */

  /* Linearized, restricted views of A and B for better alias analysis. */
  DATA_TYPE *restrict A_ = &A[0][0];
  DATA_TYPE *restrict B_ = &B[0][0];

  /* Leading dimensions for the row-major matrices. */
  const int lda = m;
  const int ldb = n;

  /* Tiling factors for k (rows) and j (columns). These values are
     chosen as reasonable defaults for modern x64 CPUs; they can be
     tuned further if desired. */
  const int T_K = 64;
  const int T_J = 64;

  const int Mdim = _PB_M;
  const int Ndim = _PB_N;

  /* Make alpha explicit loop-invariant. */
  const DATA_TYPE alpha_local = alpha;

//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM (unit lower-triangular)
// B is MxN
#pragma scop
  /* -------------------------------------------------------------- */
  /* Phase 1: accumulate strictly-lower-triangular contribution.   */
  /*           B[i,j] += sum_{k=i+1..M-1} A[k,i] * B[k,j]          */
  /*                                                              */
  /* Implemented in (k, i, j) order with blocking over k and j.   */
  /* -------------------------------------------------------------- */
  for (int kk = 1; kk < Mdim; kk += T_K)
  {
    int kend = kk + T_K;
    if (kend > Mdim)
      kend = Mdim;

    for (int jj = 0; jj < Ndim; jj += T_J)
    {
      int jend = jj + T_J;
      if (jend > Ndim)
        jend = Ndim;

      const int tile_width = jend - jj;

      for (k = kk; k < kend; ++k)
      {
        /* Pointer to source row B[k][jj..jend). */
        DATA_TYPE *restrict Bk = B_ + (size_t)k * (size_t)ldb + (size_t)jj;

        /* Target rows i = 0..k-1 (rows above the diagonal). */
        for (i = 0; i < k; ++i)
        {
          /* Load A[k][i] once and reuse across the inner j-loop. */
          const DATA_TYPE a_ki = A_[(size_t)k * (size_t)lda + (size_t)i];

          /* Pointer to target row segment B[i][jj..jend). */
          DATA_TYPE *restrict Bi = B_ + (size_t)i * (size_t)ldb + (size_t)jj;

          /* SAXPY-like update over the contiguous j-range. */
          #pragma GCC ivdep
          for (j = 0; j < tile_width; ++j)
          {
            Bi[j] += a_ki * Bk[j];
          }
        }
      }
    }
  }

  /* -------------------------------------------------------------- */
  /* Phase 2: scale B by alpha.                                    */
  /*           B[i,j] = alpha * B[i,j]                             */
  /*                                                              */
  /* All reads of B as a source (B[k,j] in Phase 1) are finished, */
  /* so scaling now is equivalent to the original per-element     */
  /* scaling but improves vectorization opportunities.            */
  /* -------------------------------------------------------------- */
  for (i = 0; i < Mdim; ++i)
  {
    DATA_TYPE *restrict Bi = B_ + (size_t)i * (size_t)ldb;

    #pragma GCC ivdep
    for (j = 0; j < Ndim; ++j)
    {
      Bi[j] *= alpha_local;
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm (m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}