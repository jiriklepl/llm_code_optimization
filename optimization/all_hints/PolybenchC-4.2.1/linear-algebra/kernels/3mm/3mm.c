/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"


/* Array initialization.
 *
 * The arithmetic structure is unchanged, but we precompute the
 * scaling factors 1/(5*ni), 1/(5*nj), ... once to avoid a division
 * in every inner-loop iteration. Multiplication by a precomputed
 * reciprocal is mathematically equivalent and keeps the same order
 * of operations for the integer arithmetic and the modulus.
 */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  /* Precompute inverse denominators to replace divisions with
     multiplications in the inner loops. */
  const DATA_TYPE inv_ni = SCALAR_VAL(1.0) / (5 * ni);
  const DATA_TYPE inv_nj = SCALAR_VAL(1.0) / (5 * nj);
  const DATA_TYPE inv_nl = SCALAR_VAL(1.0) / (5 * nl);
  const DATA_TYPE inv_nk = SCALAR_VAL(1.0) / (5 * nk);

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE)((i * j + 1) % ni) * inv_ni;

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE)((i * (j + 1) + 2) % nj) * inv_nj;

  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE)(i * (j + 3) % nl) * inv_nl;

  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE)((i * (j + 2) + 2) % nk) * inv_nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations applied:
 *  - Introduce local restrict-qualified pointers to the PolyBench
 *    arrays to help the compiler with alias analysis and vectorization.
 *  - Reorder loops for each matrix multiplication to the pattern
 *      for i
 *        zero row of result
 *        for k
 *          tmp = A[i][k];
 *          for j
 *            result[i][j] += tmp * B[k][j];
 *    This makes the innermost loop traverse contiguous memory (j is
 *    the fastest-varying index in row-major layout) for both the
 *    result matrix row and the right-hand operand row, improving
 *    cache locality and SIMD efficiency.
 *  - Keep the accumulation order over k identical to the original
 *    code for every (i,j) pair. The sequence of floating-point
 *    additions along k is unchanged, preserving numerical behavior.
 *  - Use a single OpenMP parallel region with three work-sharing
 *    loops, one per matrix multiplication. This exploits the
 *    available cores while avoiding repeated thread creation.
 *    When compiled without OpenMP support, the pragmas are ignored
 *    by the compiler and the code falls back to sequential execution.
 */
static
void kernel_3mm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		const DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		const DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		const DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		const DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

  /* Local restrict-qualified views of the arrays.
   * These do not change semantics but provide the compiler with
   * stronger aliasing guarantees, which can substantially help
   * optimization and vectorization.
   */
  DATA_TYPE (*restrict E_)[nj]       = E;
  const DATA_TYPE (*restrict A_)[nk] = A;
  const DATA_TYPE (*restrict B_)[nj] = B;
  DATA_TYPE (*restrict F_)[nl]       = F;
  const DATA_TYPE (*restrict C_)[nm] = C;
  const DATA_TYPE (*restrict D_)[nl] = D;
  DATA_TYPE (*restrict G_)[nl]       = G;

#pragma scop

  /* Single parallel region to amortize thread creation overhead. */
#pragma omp parallel private(i, j, k)
  {
    /* E := A * B
     *
     * Original:
     *   for i
     *     for j
     *       E[i][j] = 0;
     *       for k
     *         E[i][j] += A[i][k] * B[k][j];
     *
     * New ordering (i-k-j) with j innermost:
     *   for i
     *     zero E[i][:]
     *     for k
     *       for j
     *         E[i][j] += A[i][k] * B[k][j];
     *
     * For each fixed (i,j), the sequence of k-values contributing
     * to E[i][j] is still 0,1,2,...,_PB_NK-1, so the accumulation
     * order is unchanged.
     */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NI; i++)
    {
      /* Initialize row i of E. */
      for (j = 0; j < _PB_NJ; j++)
        E_[i][j] = SCALAR_VAL(0.0);

      /* Accumulate A[i][k] * B[k][:] into E[i][:]. */
      for (k = 0; k < _PB_NK; ++k)
      {
        const DATA_TYPE aik = A_[i][k];
        const DATA_TYPE *restrict Bk = B_[k];

        /* j is innermost: contiguous accesses to E_[i][j] and Bk[j]. */
        for (j = 0; j < _PB_NJ; j++)
          E_[i][j] += aik * Bk[j];
      }
    }

    /* F := C * D
     *
     * Same optimization strategy as above.
     */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NJ; i++)
    {
      /* Initialize row i of F. */
      for (j = 0; j < _PB_NL; j++)
        F_[i][j] = SCALAR_VAL(0.0);

      for (k = 0; k < _PB_NM; ++k)
      {
        const DATA_TYPE cik = C_[i][k];
        const DATA_TYPE *restrict Dk = D_[k];

        for (j = 0; j < _PB_NL; j++)
          F_[i][j] += cik * Dk[j];
      }
    }

    /* G := E * F
     *
     * Again use i-k-j ordering to keep j as innermost for
     * contiguous accesses.
     */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NI; i++)
    {
      /* Initialize row i of G. */
      for (j = 0; j < _PB_NL; j++)
        G_[i][j] = SCALAR_VAL(0.0);

      for (k = 0; k < _PB_NJ; ++k)
      {
        const DATA_TYPE eik = E_[i][k];
        const DATA_TYPE *restrict Fk = F_[k];

        for (j = 0; j < _PB_NL; j++)
          G_[i][j] += eik * Fk[j];
      }
    }
  } /* end parallel region */

#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}