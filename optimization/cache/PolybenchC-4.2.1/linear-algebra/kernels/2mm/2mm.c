/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"

/* -------------------------------------------------------------------------
 * Tunable blocking factors for the two matrix multiplications.
 *
 * These may be overridden at compile time, e.g.:
 *   -DT1_I_BLOCK=64 -DT1_J_BLOCK=64 -DT1_K_BLOCK=32
 * -------------------------------------------------------------------------*/
#ifndef T1_I_BLOCK
# define T1_I_BLOCK 32
#endif

#ifndef T1_J_BLOCK
# define T1_J_BLOCK 32
#endif

#ifndef T1_K_BLOCK
# define T1_K_BLOCK 32
#endif

#ifndef T2_I_BLOCK
# define T2_I_BLOCK 32
#endif

#ifndef T2_J_BLOCK
# define T2_J_BLOCK 32
#endif

#ifndef T2_K_BLOCK
# define T2_K_BLOCK 32
#endif


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

  /* Create restrict-qualified VLA pointers to give the compiler
   * precise aliasing information and preserve run-time dimensions. */
  DATA_TYPE (* __restrict tmp_)[nj] = tmp;
  DATA_TYPE (* __restrict A_)[nk]   = A;
  DATA_TYPE (* __restrict B_)[nj]   = B;
  DATA_TYPE (* __restrict C_)[nl]   = C;
  DATA_TYPE (* __restrict D_)[nl]   = D;

#pragma scop
  /* D := alpha*A*B*C + beta*D
   *
   * We transform the two matrix multiplications to improve data
   * locality and enable better vectorization:
   *   1) Explicitly zero tmp (to keep GEMM loops as pure reductions)
   *   2) Blocked matrix multiply: tmp = alpha * A * B
   *   3) Scale D: D = beta * D
   *   4) Blocked matrix multiply: D += tmp * C
   */

  /* 1) Initialize tmp to zero. */
  const DATA_TYPE zero = SCALAR_VAL(0.0);
  for (i = 0; i < _PB_NI; i++) {
    DATA_TYPE * __restrict tmp_row = tmp_[i];
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (j = 0; j < _PB_NJ; j++) {
      tmp_row[j] = zero;
    }
  }

  /* 2) tmp = alpha * A * B
   *
   * Blocked matrix multiplication with loop order:
   *   kk (K block), ii (I block), jj (J block), k, i, j
   *
   * For each tile, the innermost j-loop walks contiguous memory
   * in both tmp and B.  B blocks are reused across multiple rows
   * of A (i-dimension) within the tile, improving cache locality.
   */
  for (int kk = 0; kk < _PB_NK; kk += T1_K_BLOCK) {
    const int k_max = (kk + T1_K_BLOCK < _PB_NK) ? (kk + T1_K_BLOCK) : _PB_NK;

    for (int ii = 0; ii < _PB_NI; ii += T1_I_BLOCK) {
      const int i_max = (ii + T1_I_BLOCK < _PB_NI) ? (ii + T1_I_BLOCK) : _PB_NI;

      for (int jj = 0; jj < _PB_NJ; jj += T1_J_BLOCK) {
        const int j_max = (jj + T1_J_BLOCK < _PB_NJ) ? (jj + T1_J_BLOCK) : _PB_NJ;
        const int j_block_size = j_max - jj;

        for (k = kk; k < k_max; ++k) {
          const DATA_TYPE * __restrict B_row = &B_[k][jj];

          for (i = ii; i < i_max; ++i) {
            const DATA_TYPE a_ik = alpha * A_[i][k];
            DATA_TYPE * __restrict tmp_row = &tmp_[i][jj];
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (j = 0; j < j_block_size; ++j) {
              tmp_row[j] += a_ik * B_row[j];
            }
          }
        }
      }
    }
  }

  /* 3) Scale D by beta: D = beta * D.
   * This is done as a separate pass so that the accumulation in the
   * second GEMM is a pure sum, which is friendlier to vectorization. */
  if (beta != SCALAR_VAL(1.0)) {
    for (i = 0; i < _PB_NI; i++) {
      DATA_TYPE * __restrict D_row = D_[i];
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
      for (j = 0; j < _PB_NL; j++) {
        D_row[j] *= beta;
      }
    }
  }

  /* 4) D += tmp * C
   *
   * Blocked matrix multiplication with similar structure:
   *   kk (K block), ii (I block), jj (J block), k, i, j
   *
   * Here:
   *   - tmp is NI x NJ
   *   - C   is NJ x NL
   *   - D   is NI x NL
   *
   * For each tile, the innermost j-loop touches contiguous data in
   * both C and D, and C blocks are reused across multiple rows of tmp.
   */
  for (int kk = 0; kk < _PB_NJ; kk += T2_K_BLOCK) {
    const int k_max = (kk + T2_K_BLOCK < _PB_NJ) ? (kk + T2_K_BLOCK) : _PB_NJ;

    for (int ii = 0; ii < _PB_NI; ii += T2_I_BLOCK) {
      const int i_max = (ii + T2_I_BLOCK < _PB_NI) ? (ii + T2_I_BLOCK) : _PB_NI;

      for (int jj = 0; jj < _PB_NL; jj += T2_J_BLOCK) {
        const int j_max = (jj + T2_J_BLOCK < _PB_NL) ? (jj + T2_J_BLOCK) : _PB_NL;
        const int j_block_size = j_max - jj;

        for (k = kk; k < k_max; ++k) {
          const DATA_TYPE * __restrict C_row = &C_[k][jj];

          for (i = ii; i < i_max; ++i) {
            const DATA_TYPE tmp_ik = tmp_[i][k];
            DATA_TYPE * __restrict D_row = &D_[i][jj];
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (j = 0; j < j_block_size; ++j) {
              D_row[j] += tmp_ik * C_row[j];
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
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
	      alpha, beta,
	      POLYBENCH_ARRAY(tmp),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}