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


/* Array initialization.
 *
 * Optimizations:
 *  - Precompute denominators as DATA_TYPE to avoid repeated int->float
 *    conversions inside inner loops.
 *  - Use per-row pointers to improve locality and reduce address arithmetic.
 *  - Preserve the original mathematical expressions exactly.
 */
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

  *alpha = SCALAR_VAL(1.5);
  *beta  = SCALAR_VAL(1.2);

  const DATA_TYPE ni_d = (DATA_TYPE)ni;
  const DATA_TYPE nj_d = (DATA_TYPE)nj;
  const DATA_TYPE nk_d = (DATA_TYPE)nk;
  const DATA_TYPE nl_d = (DATA_TYPE)nl;

  /* A[i][j] = ((i*j+1) % ni) / ni */
  for (i = 0; i < ni; i++)
  {
    DATA_TYPE *restrict A_i = A[i];
    for (j = 0; j < nk; j++)
      A_i[j] = (DATA_TYPE)((i * j + 1) % ni) / ni_d;
  }

  /* B[i][j] = (i*(j+1) % nj) / nj */
  for (i = 0; i < nk; i++)
  {
    DATA_TYPE *restrict B_i = B[i];
    for (j = 0; j < nj; j++)
      B_i[j] = (DATA_TYPE)(i * (j + 1) % nj) / nj_d;
  }

  /* C[i][j] = ((i*(j+3)+1) % nl) / nl */
  for (i = 0; i < nj; i++)
  {
    DATA_TYPE *restrict C_i = C[i];
    for (j = 0; j < nl; j++)
      C_i[j] = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl_d;
  }

  /* D[i][j] = (i*(j+2) % nk) / nk */
  for (i = 0; i < ni; i++)
  {
    DATA_TYPE *restrict D_i = D[i];
    for (j = 0; j < nl; j++)
      D_i[j] = (DATA_TYPE)(i * (j + 2) % nk) / nk_d;
  }
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
  {
    DATA_TYPE *restrict D_i = D[i];
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D_i[j]);
    }
  }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations in kernel_2mm:
 *  - Reorder loops to make the innermost loop iterate over the last (fastest-
 *    varying) array dimension, improving spatial locality and vectorization.
 *  - Use per-row pointers (marked restrict) to reduce address computation and
 *    to help the compiler with alias analysis.
 *  - Keep, for every (i,j) element, the accumulation over k in the same
 *    increasing k order as in the original code. This preserves the original
 *    floating-point evaluation order for each matrix element.
 */
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

#pragma scop
  /* D := alpha*A*B*C + beta*D */

  /* ----------------------------------------------------------------------- */
  /* First phase: tmp = alpha * A * B                                       */
  /* ----------------------------------------------------------------------- */
  /*
   * Original code:
   *   for i
   *     for j
   *       tmp[i][j] = 0;
   *       for k
   *         tmp[i][j] += alpha * A[i][k] * B[k][j];
   *
   * New structure:
   *   - Initialize one full row tmp[i][:] to zero.
   *   - For each k, update the entire row tmp[i][:] using B[k][:].
   *   - Inner-most loop is over j, giving contiguous access to tmp[i][j] and
   *     B[k][j], which improves cache locality and vectorization.
   *   - For each fixed (i,j), the accumulation over k is still performed in
   *     the order k = 0,1,...,_PB_NK-1, matching the original evaluation order.
   */

  for (i = 0; i < _PB_NI; ++i)
  {
    DATA_TYPE *restrict tmp_i = tmp[i];
    DATA_TYPE *restrict A_i   = A[i];

    /* Initialize row i of tmp to zero. */
    for (j = 0; j < _PB_NJ; ++j)
      tmp_i[j] = SCALAR_VAL(0.0);

    /* tmp_i[j] += alpha * A_i[k] * B[k][j] */
    for (k = 0; k < _PB_NK; ++k)
    {
      const DATA_TYPE a_ik = alpha * A_i[k];
      DATA_TYPE *restrict B_k = B[k];

      /* j is innermost: contiguous accesses to B_k[j] and tmp_i[j]. */
      for (j = 0; j < _PB_NJ; ++j)
      {
        tmp_i[j] += a_ik * B_k[j];
      }
    }
  }

  /* ----------------------------------------------------------------------- */
  /* Second phase: D = beta * D + tmp * C                                   */
  /* ----------------------------------------------------------------------- */
  /*
   * Original code:
   *   for i
   *     for j
   *       D[i][j] *= beta;
   *       for k
   *         D[i][j] += tmp[i][k] * C[k][j];
   *
   * New structure:
   *   - First scale the entire row D[i][:] by beta.
   *   - Then, for each k, add tmp[i][k] * C[k][:] into D[i][:].
   *   - j is again the innermost loop, so C[k][j] and D[i][j] are accessed
   *     contiguously.
   *   - For each fixed (i,j), the accumulation over k uses the same order
   *     k = 0,1,...,_PB_NJ-1 as in the original code.
   */

  for (i = 0; i < _PB_NI; ++i)
  {
    DATA_TYPE *restrict D_i   = D[i];
    DATA_TYPE *restrict tmp_i = tmp[i];

    /* Scale row D[i][:] by beta. */
    for (j = 0; j < _PB_NL; ++j)
      D_i[j] *= beta;

    /* Accumulate the matrix product tmp * C into row D_i. */
    for (k = 0; k < _PB_NJ; ++k)
    {
      const DATA_TYPE tmp_ik = tmp_i[k];
      DATA_TYPE *restrict C_k = C[k];

      /* j is innermost: contiguous accesses to C_k[j] and D_i[j]. */
      for (j = 0; j < _PB_NL; ++j)
      {
        D_i[j] += tmp_ik * C_k[j];
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