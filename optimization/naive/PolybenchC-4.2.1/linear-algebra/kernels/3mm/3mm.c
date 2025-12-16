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


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations:
   - Change loop order from (i,j,k) to (i,k,j) for each matrix
     multiplication. The innermost j-loop now traverses rows
     of both the right-hand matrix and the result with unit
     stride, greatly improving cache locality and enabling
     straightforward auto-vectorization.
   - Split zero-initialization of E, F, G from the accumulation
     loops so that inner loops only perform fused multiply-add
     work, avoiding redundant stores.
   - Use local row pointers (marked restrict) to reduce index
     arithmetic and help the compiler reason about aliasing.
   - Cache PolyBench loop bounds (_PB_*) into local const ints
     to avoid recomputing macros and keep the original semantics.
   - For each output element, the sequence of partial sums over
     the reduction dimension (k) is preserved (still k = 0..N-1),
     so per-element floating-point accumulation order is unchanged.
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

  /* Cache PolyBench loop bounds locally (semantics identical). */
  const int ni0 = _PB_NI;
  const int nj0 = _PB_NJ;
  const int nk0 = _PB_NK;
  const int nl0 = _PB_NL;
  const int nm0 = _PB_NM;

#pragma scop
  /* ----------------------------------------------------------
   * E := A*B
   * A is (ni0 x nk0), B is (nk0 x nj0), E is (ni0 x nj0)
   * --------------------------------------------------------*/

  /* Zero-initialize E. */
  for (i = 0; i < ni0; i++) {
    DATA_TYPE *restrict Ei = E[i];
    for (j = 0; j < nj0; j++) {
      Ei[j] = SCALAR_VAL(0.0);
    }
  }

  /* Accumulate E = A*B using i-k-j loop order.
     Inner j-loop touches Ei[j] and Bk[j] with unit stride. */
  for (i = 0; i < ni0; i++) {
    DATA_TYPE *restrict Ei = E[i]; /* Row i of E */
    for (k = 0; k < nk0; ++k) {
      const DATA_TYPE aik = A[i][k];        /* A(i,k) reused across row j */
      const DATA_TYPE *restrict Bk = B[k];  /* Row k of B */
      for (j = 0; j < nj0; ++j) {
        Ei[j] += aik * Bk[j];
      }
    }
  }

  /* ----------------------------------------------------------
   * F := C*D
   * C is (nj0 x nm0), D is (nm0 x nl0), F is (nj0 x nl0)
   * --------------------------------------------------------*/

  /* Zero-initialize F. */
  for (i = 0; i < nj0; i++) {
    DATA_TYPE *restrict Fi = F[i];
    for (j = 0; j < nl0; j++) {
      Fi[j] = SCALAR_VAL(0.0);
    }
  }

  /* Accumulate F = C*D with i-k-j order. */
  for (i = 0; i < nj0; i++) {
    DATA_TYPE *restrict Fi = F[i]; /* Row i of F */
    for (k = 0; k < nm0; ++k) {
      const DATA_TYPE cik = C[i][k];        /* C(i,k) */
      const DATA_TYPE *restrict Dk = D[k];  /* Row k of D */
      for (j = 0; j < nl0; ++j) {
        Fi[j] += cik * Dk[j];
      }
    }
  }

  /* ----------------------------------------------------------
   * G := E*F
   * E is (ni0 x nj0), F is (nj0 x nl0), G is (ni0 x nl0)
   * --------------------------------------------------------*/

  /* Zero-initialize G. */
  for (i = 0; i < ni0; i++) {
    DATA_TYPE *restrict Gi = G[i];
    for (j = 0; j < nl0; j++) {
      Gi[j] = SCALAR_VAL(0.0);
    }
  }

  /* Accumulate G = E*F with i-k-j order. */
  for (i = 0; i < ni0; i++) {
    DATA_TYPE *restrict Gi = G[i]; /* Row i of G */
    for (k = 0; k < nj0; ++k) {
      const DATA_TYPE eik = E[i][k];        /* E(i,k) */
      const DATA_TYPE *restrict Fk = F[k];  /* Row k of F */
      for (j = 0; j < nl0; ++j) {
        Gi[j] += eik * Fk[j];
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