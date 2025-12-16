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


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
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


/*
 * Main computational kernel.
 *
 * Optimizations:
 * - Loop order changed from (i, j, k) to (i, k, j) for all three GEMMs.
 *   This makes the innermost loop iterate over the last (contiguous)
 *   dimension of the matrices, improving cache locality and enabling
 *   more effective SIMD vectorization.
 *
 * - Local restrict-qualified pointers (E_, A_, B_, F_, C_, D_, G_) are
 *   introduced to express non-aliasing to the compiler, aiding
 *   auto-vectorization and other optimizations. The underlying arrays
 *   are in fact distinct in this benchmark, so this preserves semantics.
 *
 * - OpenMP parallelization over the outer “i” loop for each matrix
 *   multiplication. Each iteration of the i-loop updates disjoint rows
 *   of the output matrix, so there are no data races. When compiled
 *   without OpenMP support, the pragmas are ignored and the code
 *   behaves exactly like a sequential version.
 *
 * The algebraic computation (E = A*B, F = C*D, G = E*F) is unchanged.
 */
static
void kernel_3mm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

  /* Use PolyBench’s possibly adjusted loop bounds. */
  const int ni_ = _PB_NI;
  const int nj_ = _PB_NJ;
  const int nk_ = _PB_NK;
  const int nl_ = _PB_NL;
  const int nm_ = _PB_NM;

  /* Create local restrict-qualified views to help the optimizer.
     The second dimension uses the runtime sizes ni, nj, nk, nl, nm,
     matching the actual allocated leading dimensions. */
  DATA_TYPE (* restrict E_)[nj] = E;
  DATA_TYPE (* restrict A_)[nk] = A;
  DATA_TYPE (* restrict B_)[nj] = B;
  DATA_TYPE (* restrict F_)[nl] = F;
  DATA_TYPE (* restrict C_)[nm] = C;
  DATA_TYPE (* restrict D_)[nl] = D;
  DATA_TYPE (* restrict G_)[nl] = G;

#pragma scop
  /* E := A*B
   *
   * Original order: for i, for j, for k.
   * New order:      for i, for k, for j.
   * For each (i,j), the accumulation over k is performed in the same
   * k order as before; we only interleave different j’s, which does
   * not change the mathematical result.
   */
#pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE * restrict E_row = E_[i];

    /* Initialize the output row once. */
    for (j = 0; j < nj_; j++)
      E_row[j] = SCALAR_VAL(0.0);

    /* Accumulate A(i,k) * B(k,*) into the whole row of E in a
       cache- and SIMD-friendly way. */
    for (k = 0; k < nk_; ++k)
    {
      const DATA_TYPE a_ik = A_[i][k];
      const DATA_TYPE * restrict B_row = B_[k];

      for (j = 0; j < nj_; ++j)
        E_row[j] += a_ik * B_row[j];
    }
  }

  /* F := C*D
   *
   * Same optimization pattern as for E := A*B, adapted to dimensions.
   */
#pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < nj_; i++)
  {
    DATA_TYPE * restrict F_row = F_[i];

    for (j = 0; j < nl_; j++)
      F_row[j] = SCALAR_VAL(0.0);

    for (k = 0; k < nm_; ++k)
    {
      const DATA_TYPE c_ik = C_[i][k];
      const DATA_TYPE * restrict D_row = D_[k];

      for (j = 0; j < nl_; ++j)
        F_row[j] += c_ik * D_row[j];
    }
  }

  /* G := E*F
   *
   * Again use (i, k, j) loop order so the innermost loop traverses
   * contiguous elements of F and G.
   */
#pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE * restrict G_row = G_[i];

    for (j = 0; j < nl_; j++)
      G_row[j] = SCALAR_VAL(0.0);

    for (k = 0; k < nj_; ++k)
    {
      const DATA_TYPE e_ik = E_[i][k];
      const DATA_TYPE * restrict F_row = F_[k];

      for (j = 0; j < nl_; ++j)
        G_row[j] += e_ik * F_row[j];
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