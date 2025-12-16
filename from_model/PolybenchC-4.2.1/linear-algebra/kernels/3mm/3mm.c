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
 *
 * Optimizations applied:
 *   - Cache blocking (tiling) on all three GEMMs (E, F, G) to improve
 *     data locality in A/B/C/D/E/F/G.
 *   - Loop reordering so that the innermost loops iterate over the
 *     contiguous dimension (columns in row-major storage), exposing
 *     vectorization opportunities.
 *   - Scalar replacement: hoist A[i,k], C[j,m], E[i,j] into scalars
 *     and reuse them across the innermost loops.
 *   - Use of local restrict-qualified aliases to improve alias analysis
 *     and enable more aggressive optimization by the compiler.
 *
 * The mathematical results are identical to the original code (up to
 * floating‑point round-off due to reordering of associative sums).
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

  /* Tunable tile sizes. Values 32 work well for typical PolyBench
     problem sizes and modern caches, but can be adjusted if needed. */
  const int TI_E = 32;
  const int TJ_E = 32;
  const int TK_E = 32;

  const int TJ_F = 32;
  const int TL_F = 32;
  const int TM_F = 32;

  const int TI_G = 32;
  const int TL_G = 32;
  const int TJ_G = 32;

  /* Local restrict-qualified aliases (do not change semantics because
     the actual arrays do not alias each other). This helps the compiler
     with vectorization and common subexpression elimination. */
  DATA_TYPE (*restrict E_)[nj] = E;
  DATA_TYPE (*restrict A_)[nk] = A;
  DATA_TYPE (*restrict B_)[nj] = B;
  DATA_TYPE (*restrict F_)[nl] = F;
  DATA_TYPE (*restrict C_)[nm] = C;
  DATA_TYPE (*restrict D_)[nl] = D;
  DATA_TYPE (*restrict G_)[nl] = G;

#pragma scop
  /* --------------------------------------------------------------
   * E := A * B
   * Original:
   *   for i
   *     for j
   *       E[i][j] = 0;
   *       for k
   *         E[i][j] += A[i][k] * B[k][j];
   *
   * Optimized structure:
   *   - Separate initialization E = 0.
   *   - Tiled i-k-j loop nest with innermost j (contiguous).
   *   - Hoist A[i][k] and row pointers of E and B into scalars.
   * -------------------------------------------------------------- */

  /* Initialize E to zero. */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      E_[i][j] = SCALAR_VAL(0.0);

  /* Tiled matrix multiplication: E = A * B. */
  for (int ii = 0; ii < _PB_NI; ii += TI_E)
  {
    const int i_max = (ii + TI_E < _PB_NI) ? (ii + TI_E) : _PB_NI;

    for (int kk = 0; kk < _PB_NK; kk += TK_E)
    {
      const int k_max = (kk + TK_E < _PB_NK) ? (kk + TK_E) : _PB_NK;

      for (int jj = 0; jj < _PB_NJ; jj += TJ_E)
      {
        const int j_max   = (jj + TJ_E < _PB_NJ) ? (jj + TJ_E) : _PB_NJ;
        const int width_j = j_max - jj;

        for (i = ii; i < i_max; i++)
        {
          /* Row of E corresponding to the current i and j-tile. */
          DATA_TYPE *restrict e_row_base = &E_[i][jj];

          for (k = kk; k < k_max; k++)
          {
            const DATA_TYPE a_ik = A_[i][k];
            const DATA_TYPE *restrict b_row_base = &B_[k][jj];

            /* Innermost loop over contiguous j dimension. */
            for (int tj = 0; tj < width_j; tj++)
            {
              e_row_base[tj] += a_ik * b_row_base[tj];
            }
          }
        }
      }
    }
  }

  /* --------------------------------------------------------------
   * F := C * D
   * Original:
   *   for j
   *     for l
   *       F[j][l] = 0;
   *       for m
   *         F[j][l] += C[j][m] * D[m][l];
   *
   * Optimized structure:
   *   - Separate initialization F = 0.
   *   - Tiled j-m-l loop nest with innermost l (contiguous).
   *   - Hoist C[j][m] and row pointers of F and D into scalars.
   * -------------------------------------------------------------- */

  /* Initialize F to zero. */
  for (i = 0; i < _PB_NJ; i++)
    for (j = 0; j < _PB_NL; j++)
      F_[i][j] = SCALAR_VAL(0.0);

  /* Tiled matrix multiplication: F = C * D. */
  for (int jj = 0; jj < _PB_NJ; jj += TJ_F)
  {
    const int j_max = (jj + TJ_F < _PB_NJ) ? (jj + TJ_F) : _PB_NJ;

    for (int mm = 0; mm < _PB_NM; mm += TM_F)
    {
      const int m_max = (mm + TM_F < _PB_NM) ? (mm + TM_F) : _PB_NM;

      for (int ll = 0; ll < _PB_NL; ll += TL_F)
      {
        const int l_max   = (ll + TL_F < _PB_NL) ? (ll + TL_F) : _PB_NL;
        const int width_l = l_max - ll;

        /* Note: here loop variable 'i' iterates over the logical 'j'
           index of F and C (rows). */
        for (i = jj; i < j_max; i++)
        {
          DATA_TYPE *restrict f_row_base = &F_[i][ll];

          /* 'k' iterates over logical 'm' (columns of C, rows of D). */
          for (k = mm; k < m_max; k++)
          {
            const DATA_TYPE c_jm = C_[i][k];
            const DATA_TYPE *restrict d_row_base = &D_[k][ll];

            for (int tl = 0; tl < width_l; tl++)
            {
              f_row_base[tl] += c_jm * d_row_base[tl];
            }
          }
        }
      }
    }
  }

  /* --------------------------------------------------------------
   * G := E * F
   * Original:
   *   for i
   *     for l
   *       G[i][l] = 0;
   *       for j
   *         G[i][l] += E[i][j] * F[j][l];
   *
   * Optimized structure:
   *   - Separate initialization G = 0.
   *   - Tiled i-j-l loop nest with innermost l (contiguous).
   *   - Hoist E[i][j] and row pointers of G and F into scalars.
   * -------------------------------------------------------------- */

  /* Initialize G to zero. */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      G_[i][j] = SCALAR_VAL(0.0);

  /* Tiled matrix multiplication: G = E * F. */
  for (int ii = 0; ii < _PB_NI; ii += TI_G)
  {
    const int i_max = (ii + TI_G < _PB_NI) ? (ii + TI_G) : _PB_NI;

    for (int jj = 0; jj < _PB_NJ; jj += TJ_G)
    {
      const int j_max = (jj + TJ_G < _PB_NJ) ? (jj + TJ_G) : _PB_NJ;

      for (int ll = 0; ll < _PB_NL; ll += TL_G)
      {
        const int l_max   = (ll + TL_G < _PB_NL) ? (ll + TL_G) : _PB_NL;
        const int width_l = l_max - ll;

        for (i = ii; i < i_max; i++)
        {
          DATA_TYPE *restrict g_row_base = &G_[i][ll];

          /* 'k' here iterates over the logical 'j' reduction dimension. */
          for (k = jj; k < j_max; k++)
          {
            const DATA_TYPE e_ij = E_[i][k];
            const DATA_TYPE *restrict f_row_base = &F_[k][ll];

            for (int tl = 0; tl < width_l; tl++)
            {
              g_row_base[tl] += e_ij * f_row_base[tl];
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