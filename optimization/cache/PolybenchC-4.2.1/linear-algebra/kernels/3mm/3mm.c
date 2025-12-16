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

/* -------------------------------------------------------------------
 * Tunable tiling parameters for the three matrix multiplications.
 * These can be overridden at compile time, e.g.:
 *   gcc -O3 -DTILE_I=64 -DTILE_J=64 -DTILE_K=32 ...
 *
 * They are used to improve cache locality by operating on small
 * sub-blocks that fit into L1/L2 caches.
 * ------------------------------------------------------------------- */
#ifndef TILE_I
# define TILE_I 32
#endif

#ifndef TILE_J
# define TILE_J 32
#endif

#ifndef TILE_K
# define TILE_K 32
#endif


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

   Optimizations applied:
   - Use restrict-qualified local aliases for all matrices to help
     the compiler with alias analysis and vectorization.
   - Zero-initialize E, F and G once with memset instead of repeatedly
     assigning 0.0 inside the j-loop.
   - Reorder loops to i-k-j and apply cache blocking (tiling) for all
     three matrix multiplications. The innermost loop always iterates
     over j so that accesses to the second index (row-major contiguous)
     are unit-stride for both the left-hand side and right-hand side
     matrices.
   - The blocking factors TILE_I/J/K are tunable and chosen to improve
     data locality on typical x86_64 cache hierarchies.
   - The order of accumulation for each scalar result (over k) is
     preserved (still strictly increasing k), so floating-point
     semantics remain bit-identical to the original code.
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

  /* Local restrict-qualified aliases using the runtime sizes.
     This tells the compiler that these matrices do not alias each
     other, which enables more aggressive optimizations. */
  DATA_TYPE       (* restrict E_)[nj] = E;
  const DATA_TYPE (* restrict A_)[nk] = A;
  const DATA_TYPE (* restrict B_)[nj] = B;
  DATA_TYPE       (* restrict F_)[nl] = F;
  const DATA_TYPE (* restrict C_)[nm] = C;
  const DATA_TYPE (* restrict D_)[nl] = D;
  DATA_TYPE       (* restrict G_)[nl] = G;

#pragma scop
  /* Zero-initialize result matrices once.
     This is equivalent to the original code that set each element
     to 0.0 immediately before computing its dot product. */
  memset(E_, 0, (size_t)ni * (size_t)nj * sizeof(DATA_TYPE));
  memset(F_, 0, (size_t)nj * (size_t)nl * sizeof(DATA_TYPE));
  memset(G_, 0, (size_t)ni * (size_t)nl * sizeof(DATA_TYPE));

  /* E := A * B
     Blocked matrix multiplication.
     Loop order and tiling improve spatial and temporal locality:
       - A_ is accessed by rows (i, k).
       - B_ and E_ are accessed with unit stride in j. */
  for (int ii = 0; ii < _PB_NI; ii += TILE_I)
  {
    const int i_end = (ii + TILE_I > _PB_NI) ? _PB_NI : (ii + TILE_I);

    for (int kk = 0; kk < _PB_NK; kk += TILE_K)
    {
      const int k_end = (kk + TILE_K > _PB_NK) ? _PB_NK : (kk + TILE_K);

      for (int jj = 0; jj < _PB_NJ; jj += TILE_J)
      {
        const int j_end = (jj + TILE_J > _PB_NJ) ? _PB_NJ : (jj + TILE_J);

        for (i = ii; i < i_end; ++i)
        {
          DATA_TYPE * restrict Ei = E_[i];

          for (k = kk; k < k_end; ++k)
          {
            const DATA_TYPE aik = A_[i][k];
            const DATA_TYPE * restrict Bk = B_[k];

            /* No loop-carried dependencies across j; safe to vectorize. */
#pragma GCC ivdep
            for (j = jj; j < j_end; ++j)
            {
              Ei[j] += aik * Bk[j];
            }
          }
        }
      }
    }
  }

  /* F := C * D */
  for (int ii = 0; ii < _PB_NJ; ii += TILE_I)
  {
    const int i_end = (ii + TILE_I > _PB_NJ) ? _PB_NJ : (ii + TILE_I);

    for (int kk = 0; kk < _PB_NM; kk += TILE_K)
    {
      const int k_end = (kk + TILE_K > _PB_NM) ? _PB_NM : (kk + TILE_K);

      for (int jj = 0; jj < _PB_NL; jj += TILE_J)
      {
        const int j_end = (jj + TILE_J > _PB_NL) ? _PB_NL : (jj + TILE_J);

        for (i = ii; i < i_end; ++i)
        {
          DATA_TYPE * restrict Fi = F_[i];

          for (k = kk; k < k_end; ++k)
          {
            const DATA_TYPE cik = C_[i][k];
            const DATA_TYPE * restrict Dk = D_[k];

#pragma GCC ivdep
            for (j = jj; j < j_end; ++j)
            {
              Fi[j] += cik * Dk[j];
            }
          }
        }
      }
    }
  }

  /* G := E * F */
  for (int ii = 0; ii < _PB_NI; ii += TILE_I)
  {
    const int i_end = (ii + TILE_I > _PB_NI) ? _PB_NI : (ii + TILE_I);

    for (int kk = 0; kk < _PB_NJ; kk += TILE_K)
    {
      const int k_end = (kk + TILE_K > _PB_NJ) ? _PB_NJ : (kk + TILE_K);

      for (int jj = 0; jj < _PB_NL; jj += TILE_J)
      {
        const int j_end = (jj + TILE_J > _PB_NL) ? _PB_NL : (jj + TILE_J);

        for (i = ii; i < i_end; ++i)
        {
          DATA_TYPE * restrict Gi = G_[i];

          for (k = kk; k < k_end; ++k)
          {
            const DATA_TYPE eik = E_[i][k];
            const DATA_TYPE * restrict Fk = F_[k];

#pragma GCC ivdep
            for (j = jj; j < j_end; ++j)
            {
              Gi[j] += eik * Fk[j];
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