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
 * Optimized GEMM implementation.
 *
 * Key changes compared to the original kernel:
 * 1. Use `restrict`-qualified local pointers for A, B, C to help the
 *    compiler assume no aliasing and generate better code.
 * 2. Separate the scaling of C by `beta` into its own loop nest.
 *    This improves temporal locality and simplifies the main GEMM loops.
 * 3. Use cache-blocked (tiled) matrix multiplication:
 *       C += alpha * A * B
 *    with blocking in i, j, and k dimensions to improve data reuse and
 *    cache locality, while preserving the mathematical result.
 * 4. Keep the innermost loop over j (row-major contiguous dimension)
 *    to enable efficient vectorization.
 *
 * Semantics are preserved:
 *   C_final = beta * C_initial + alpha * A * B
 * ---------------------------------------------------------------------*/


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

  /* Create restrict-qualified local views of the 2D arrays.
   * This tells the compiler that A, B, and C do not alias, which
   * is true for this benchmark and greatly helps vectorization. */
  DATA_TYPE (*restrict C_)[nj] = C;
  DATA_TYPE (*restrict A_)[nk] = A;
  DATA_TYPE (*restrict B_)[nj] = B;

  /* Cache the (possibly parameterized) problem sizes used in loops. */
  const int ni2 = _PB_NI;
  const int nj2 = _PB_NJ;
  const int nk2 = _PB_NK;

  /* Blocking factors: chosen to fit well into L1/L2 caches for
   * double-precision data on typical x64 machines.
   * They do not need to divide ni/nj/nk exactly; edge blocks are
   * handled via min()-style bounds. */
  const int Ti = 32;
  const int Tj = 32;
  const int Tk = 32;

  //BLAS PARAMS
  //TRANSA = 'N'
  //TRANSB = 'N'
  // => Form C := alpha*A*B + beta*C,
  //A is NIxNK
  //B is NKxNJ
  //C is NIxNJ
#pragma scop

  /* ------------------------------------------------------------------ */
  /* 1. Scale C by beta: C = beta * C                                   */
  /* ------------------------------------------------------------------ */
  for (i = 0; i < ni2; ++i) {
    DATA_TYPE *restrict Ci = &C_[i][0];
    for (j = 0; j < nj2; ++j) {
      Ci[j] *= beta;
    }
  }

  /* ------------------------------------------------------------------ */
  /* 2. Blocked GEMM: C += alpha * A * B                                */
  /*    Loop order: (ii, kk, jj) outer tiles, then (i, k, j) inside.    */
  /*    - i, j index C rows/cols (row-major storage).                   */
  /*    - k walks the shared dimension.                                 */
  /*    - j is the innermost loop for contiguous memory & SIMD.         */
  /* ------------------------------------------------------------------ */
  for (int ii = 0; ii < ni2; ii += Ti) {
    const int i_end = (ii + Ti < ni2) ? (ii + Ti) : ni2;

    for (int kk = 0; kk < nk2; kk += Tk) {
      const int k_end = (kk + Tk < nk2) ? (kk + Tk) : nk2;

      for (int jj = 0; jj < nj2; jj += Tj) {
        const int j_end = (jj + Tj < nj2) ? (jj + Tj) : nj2;
        const int jb    = j_end - jj; /* block width in j */

        for (i = ii; i < i_end; ++i) {
          DATA_TYPE *restrict Ci_block       = &C_[i][jj];
          const DATA_TYPE *restrict Ai_row   = &A_[i][0];

          for (k = kk; k < k_end; ++k) {
            const DATA_TYPE a_ik = alpha * Ai_row[k];
            const DATA_TYPE *restrict Bk_block = &B_[k][jj];

            /* Inner-most loop over contiguous j range: vector-friendly. */
            for (j = 0; j < jb; ++j) {
              Ci_block[j] += a_ik * Bk_block[j];
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