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

/* ---------------------------------------------------------------------
 * Tunable blocking parameters.
 *
 * These can be adjusted to better fit a particular cache hierarchy.
 * They do not need to divide NI/NJ/NK; edge cases are handled explicitly.
 * ------------------------------------------------------------------- */
#ifndef GEMM_BLOCK_I
# define GEMM_BLOCK_I 64
#endif

#ifndef GEMM_BLOCK_J
# define GEMM_BLOCK_J 64
#endif

#ifndef GEMM_BLOCK_K
# define GEMM_BLOCK_K 64
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
   including the call and return.
 *
 * Optimizations applied:
 *  - Explicit separation of the beta scaling from the GEMM update.
 *    This exposes a pure matrix-multiply to the optimizer.
 *  - Blocking (tiling) over i, j, k to improve cache locality.
 *    Each tile operates on small C and B submatrices that fit better in cache.
 *  - Use of restrict-qualified local pointers to help the compiler
 *    assume non-aliasing and generate better code (including vectorization).
 *  - Optional OpenMP pragmas to exploit thread-level parallelism when
 *    compiled with -fopenmp. When OpenMP is disabled, pragmas are ignored
 *    and the code remains valid and sequential.
 *
 * Floating-point semantics:
 *  - For each element C[i][j], the order of accumulation over k is preserved:
 *      scale by beta, then accumulate for k = 0..nk-1 in increasing order.
 *    We only change the order in which *different* elements of C are updated,
 *    which does not affect their individual values.
 */
static
void __attribute__((noinline))
kernel_gemm(int ni, int nj, int nk,
	    DATA_TYPE alpha,
	    DATA_TYPE beta,
	    DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
	    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
	    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

  /* Create local restrict-qualified views of the 2D arrays.
   * Using VLAs here ensures we exactly match the runtime sizes (ni, nj, nk),
   * while 'restrict' tells the compiler that A, B, and C do not alias. */
  DATA_TYPE (*restrict C_)[nj] = C;
  DATA_TYPE (*restrict A_)[nk] = A;
  DATA_TYPE (*restrict B_)[nj] = B;

  /* Effective problem sizes.  We keep using the runtime sizes (ni, nj, nk)
   * rather than the compile-time NI/NJ/NK, which is safe because main()
   * passes matching dimensions when allocating the arrays. */
  const int ni_eff = ni;
  const int nj_eff = nj;
  const int nk_eff = nk;

  const int BI = GEMM_BLOCK_I;
  const int BJ = GEMM_BLOCK_J;
  const int BK = GEMM_BLOCK_K;

  /* BLAS PARAMS
   * TRANSA = 'N'
   * TRANSB = 'N'
   * => Form C := alpha*A*B + beta*C,
   * A is NIxNK
   * B is NKxNJ
   * C is NIxNJ
   */

#pragma scop

  /* ------------------------------------------------------------------ */
  /* 1. Scale C by beta: C[i][j] = beta * C[i][j]                       */
  /*    This is separated from the GEMM update to simplify the main    */
  /*    kernel and allow independent optimization / parallelization.   */
  /* ------------------------------------------------------------------ */
  /* Parallel across the two outer dimensions when OpenMP is enabled.  */
#pragma omp parallel for collapse(2) schedule(static) if (ni_eff * nj_eff > 1024)
  for (i = 0; i < ni_eff; i++) {
    for (j = 0; j < nj_eff; j++) {
      C_[i][j] *= beta;
    }
  }

  /* ------------------------------------------------------------------ */
  /* 2. Blocked matrix multiplication: C += alpha * A * B               */
  /*    We tile the iteration space over i, j, and k. The tiling order  */
  /*    and inner loops are chosen so that:                             */
  /*      - The innermost loop iterates over j, giving unit-stride      */
  /*        access for both B[k][j] and C[i][j] (row-major layout).     */
  /*      - A[i][k] is reused across the j-loop via a scalar register.  */
  /*      - B and C tiles are kept as small contiguous blocks in cache. */
  /* ------------------------------------------------------------------ */

  /* Parallelize over the i- and j-blocks. Each (ii,jj) tile owns a
   * disjoint submatrix of C, so no synchronization is required.        */
#pragma omp parallel for collapse(2) schedule(static) private(i, j, k) if (ni_eff * nj_eff > 1024)
  for (int ii = 0; ii < ni_eff; ii += BI) {
    for (int jj = 0; jj < nj_eff; jj += BJ) {

      /* Compute actual block boundaries (handle edge tiles). */
      const int i_end = (ii + BI < ni_eff) ? (ii + BI) : ni_eff;
      const int j_end = (jj + BJ < nj_eff) ? (jj + BJ) : nj_eff;

      for (int kk = 0; kk < nk_eff; kk += BK) {
        const int k_end = (kk + BK < nk_eff) ? (kk + BK) : nk_eff;

        for (i = ii; i < i_end; ++i) {
          /* For each row i in the current tile, walk through a block
           * of k and accumulate into the corresponding C row. */
          for (k = kk; k < k_end; ++k) {
            /* Pre-scale A[i][k] by alpha once and reuse across j. */
            const DATA_TYPE a_ik = alpha * A_[i][k];

            /* Local pointers to the current row-blocks of C and B.
             * This keeps indexing simple and encourages vectorization. */
            DATA_TYPE *restrict c_block = &C_[i][jj];
            const DATA_TYPE *restrict b_block = &B_[k][jj];

            const int width = j_end - jj;

            /* Innermost j-loop: contiguous access in memory.
             * The compiler can typically auto-vectorize this loop.
             */
#pragma omp simd
            for (int tj = 0; tj < width; ++tj) {
              c_block[tj] += a_ik * b_block[tj];
            }
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