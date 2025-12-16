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


/* ---------------------------------------------------------------------------
 * Tunable blocking parameters.
 *
 * These control the cache blocking of the GEMM kernel. They can be
 * overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DGEMM_BLOCK_I=64 -DGEMM_BLOCK_J=64 -DGEMM_BLOCK_K=32 ...
 *
 * Default values are chosen so that, for both float and double, a
 * BLOCK_I x BLOCK_J C tile together with the corresponding A and B
 * panels fit comfortably in L1 cache.
 * ------------------------------------------------------------------------- */
#ifndef GEMM_BLOCK_I
# define GEMM_BLOCK_I 32
#endif

#ifndef GEMM_BLOCK_J
# define GEMM_BLOCK_J 32
#endif

#ifndef GEMM_BLOCK_K
# define GEMM_BLOCK_K 32
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
  /* Use local constants for PolyBench loop bounds. This avoids
   * re-evaluating the POLYBENCH_LOOP_BOUND macro in every loop header. */
  const int ni_pb = _PB_NI;
  const int nj_pb = _PB_NJ;
  const int nk_pb = _PB_NK;

  /* Local copies of scalars encourage register promotion. */
  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

  /* Local copies of block sizes allow the compiler to treat them as
   * regular integers (and potentially unroll for common values). */
  const int BI = GEMM_BLOCK_I;
  const int BJ = GEMM_BLOCK_J;
  const int BK = GEMM_BLOCK_K;

  int ii, jj, i, j, kk, k;

  /* BLAS PARAMS
   * TRANSA = 'N'
   * TRANSB = 'N'
   * => Form C := alpha*A*B + beta*C,
   * A is NIxNK
   * B is NKxNJ
   * C is NIxNJ
   *
   * Optimized implementation:
   * -------------------------
   * - We block the i, j and k loops to improve cache locality.
   *   Each (ii,jj) tile defines a sub-block of C of size at most
   *   BI x BJ, which fits well in L1 cache together with the
   *   corresponding panels of A and B.
   *
   * - For each C tile, we:
   *     (1) Scale C by beta_local once.
   *     (2) Accumulate alpha_local * A * B over the full k-range
   *         in BK-sized panels. For any fixed (i,j), the k-loop
   *         still runs from 0 to nk_pb-1 in increasing order,
   *         preserving the original floating-point evaluation
   *         order with respect to k.
   *
   * - The outermost two loops over (ii,jj) are parallelized with
   *   OpenMP. Each thread operates on a disjoint tile of C, so there
   *   are no data races. A and B are read-only during the kernel.
   *
   *   If compiled without OpenMP support, the '#pragma omp' directive
   *   is ignored by the compiler and the code remains correct.
   */

#pragma scop
  /* Parallelize over (ii,jj) tiles. Each tile updates a unique
   * sub-block of C, so there is no cross-thread interference. */
#pragma omp parallel for collapse(2) schedule(static) private(ii, jj, i, j, kk, k)
  for (ii = 0; ii < ni_pb; ii += BI)
  {
    for (jj = 0; jj < nj_pb; jj += BJ)
    {
      const int i_end = (ii + BI < ni_pb) ? (ii + BI) : ni_pb;
      const int j_end = (jj + BJ < nj_pb) ? (jj + BJ) : nj_pb;
      const int jb = j_end - jj; /* actual width of this tile */

      /* Process each row i of the current (ii,jj) tile. */
      for (i = ii; i < i_end; ++i)
      {
        /* Pointer to the first element of C(i, jj:jj+jb-1). Using
         * explicit pointers reduces address calculation overhead
         * inside the innermost loops. */
        DATA_TYPE *restrict Ci = &C[i][jj];

        /* 1. Scale C(i, jj:jj+jb-1) by beta_local exactly once. */
        for (j = 0; j < jb; ++j)
          Ci[j] *= beta_local;

        /* 2. Accumulate alpha_local * A(i, :) * B(:, jj:jj+jb-1)
         *    over the full k range, blocked by BK. */
        for (kk = 0; kk < nk_pb; kk += BK)
        {
          const int k_end = (kk + BK < nk_pb) ? (kk + BK) : nk_pb;

          for (k = kk; k < k_end; ++k)
          {
            /* Load A(i,k) once and scale by alpha_local. This value
             * is reused across the entire j-loop. */
            const DATA_TYPE a_ik = alpha_local * A[i][k];

            /* Pointer to the first element of B(k, jj:jj+jb-1). */
            const DATA_TYPE *restrict Bk = &B[k][jj];

            /* Innermost loop: update the C tile row with a rank-1
             * outer product contribution. Accesses to Ci[j] and
             * Bk[j] are contiguous and thus cache-friendly and
             * vectorizable. */
            for (j = 0; j < jb; ++j)
              Ci[j] += a_ik * Bk[j];
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