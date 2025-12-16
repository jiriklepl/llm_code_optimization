/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

/* --------------------------------------------------------------------
 * Tunable blocking parameters.
 *
 * These control the cache blocking of the GEMM kernel. They can be
 * overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DGEMM_BLOCK_J=64 -DGEMM_BLOCK_K=128 ...
 *
 * The defaults are chosen for typical x86-64 systems with 32–64KB L1.
 * ------------------------------------------------------------------ */
#ifndef GEMM_BLOCK_J
#  define GEMM_BLOCK_J 64
#endif

#ifndef GEMM_BLOCK_K
#  define GEMM_BLOCK_K 64
#endif


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

  /* Create local restricted views of the 2D arrays.
   *
   * - Using 'restrict' tells the compiler that A, B, and C do not alias,
   *   which helps vectorization and other aggressive optimizations.
   * - Keeping the second dimension explicit preserves row-major layout
   *   information so the compiler can generate unit-stride accesses.
   */
  int ni0 = ni;
  int nj0 = nj;
  int nk0 = nk;

  DATA_TYPE (* restrict c)[nj0] = C;
  DATA_TYPE (* restrict a)[nk0] = A;
  DATA_TYPE (* restrict b)[nj0] = B;

  const int block_j = GEMM_BLOCK_J;
  const int block_k = GEMM_BLOCK_K;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  /* Outer loop over rows of C and A.
   *
   * For each row i:
   *   1) Scale C[i][:] by beta (unit-stride over j).
   *   2) Perform a blocked update C[i][:] += alpha * A[i][:] * B[:][:].
   *
   * Loop transformations are chosen to:
   *   - keep j as the innermost index to get unit-stride accesses in B[k][j]
   *     and C[i][j];
   *   - block over k and j to improve cache reuse of B and C;
   *   - maintain the original accumulation order over k for each C[i][j].
   */
  for (i = 0; i < _PB_NI; i++) {
    DATA_TYPE * restrict ci = c[i];
    DATA_TYPE * restrict ai = a[i];

    /* 1. Scale C's i-th row by beta.
     *    This is identical to the original C[i][j] *= beta;
     *    We only rearrange pointer usage for better optimization.
     */
    for (j = 0; j < _PB_NJ; j++)
      ci[j] *= beta;

    /* 2. Blocked matrix multiplication for row i:
     *
     * Original order for the update was:
     *   for (k = 0; k < _PB_NK; k++)
     *     for (j = 0; j < _PB_NJ; j++)
     *       C[i][j] += alpha * A[i][k] * B[k][j];
     *
     * We keep the logical (k, j) ordering but iterate over (k, j)
     * in tiles (kk0..kend-1) x (jj0..jend-1). This preserves, for each
     * (i, j), the order in which the partial sums over k are accumulated
     * (0.._PB_NK-1), thereby preserving the original floating-point
     * accumulation semantics while improving cache locality.
     */
    for (int kk0 = 0; kk0 < _PB_NK; kk0 += block_k) {
      const int kend = (kk0 + block_k < _PB_NK) ? (kk0 + block_k) : _PB_NK;

      for (int jj0 = 0; jj0 < _PB_NJ; jj0 += block_j) {
        const int jend = (jj0 + block_j < _PB_NJ) ? (jj0 + block_j) : _PB_NJ;

        /* Iterate over the current k-block. Each A[i][k] is loaded once
         * per tile of C and reused across the inner j-loop, which walks
         * contiguous elements of both B[k][:] and C[i][:].
         */
        for (k = kk0; k < kend; k++) {
          const DATA_TYPE aik = alpha * ai[k];
          const DATA_TYPE * restrict bk = b[k];

          /* Innermost loop: contiguous access in j enables efficient
           * vectorization (SIMD) and good cache behavior.
           */
          for (j = jj0; j < jend; j++) {
            ci[j] += aik * bk[j];
          }
        }
      }
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}