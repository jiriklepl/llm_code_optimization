/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"

/* -------------------------------------------------------------------------
 * Tunable parameters
 * -------------------------------------------------------------------------
 *
 * DOITGEN_TILE_P controls blocking in the p-dimension of the kernel.
 * It can be overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DDOITGEN_TILE_P=128 ...
 *
 * The tiling is semantics-preserving (no change in the order of
 * floating‑point additions for each output element).
 */
#ifndef DOITGEN_TILE_P
# define DOITGEN_TILE_P 64
#endif

/* Basic sanity check for the tile size. */
_Static_assert(DOITGEN_TILE_P > 0, "DOITGEN_TILE_P must be positive");


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  /* A is laid out in row-major order with the last index contiguous.
   * The original loop nest (k innermost) already walks it with
   * unit-stride accesses, so we keep that structure. */
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;

  /* C4[i][j] is also stored with j (the last index) contiguous;
   * again, keep j as the innermost loop to preserve unit stride. */
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Original mathematical operation for each (r, q):

     sum[p] = sum_{s} A[r][q][s] * C4[s][p];
     A[r][q][p] = sum[p];

   This implementation preserves the exact summation order for every
   (r, q, p) while improving data locality and enabling better
   vectorization:

   - We iterate with s as the outer accumulation loop and p as the
     inner loop so that C4[s][p] and sum[p] are accessed with
     unit stride (contiguous in memory). A[r][q][s] is also accessed
     with unit stride because s is the last index of A[r][q][s].
   - We add a tunable block over p (DOITGEN_TILE_P) to control cache
     behavior for large NP.
   - We use local restrict-qualified pointers to help the compiler
     with alias analysis and vectorization.
*/
void kernel_doitgen[[gnu::flatten, gnu::noinline]](int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++) {
    for (q = 0; q < _PB_NQ; q++)  {

      /* Local restricted pointers to help the optimizer.
       * A_rq points to the contiguous row A[r][q][0 .. NP-1].
       * sum_row is the temporary accumulation buffer of length NP. */
      DATA_TYPE *restrict A_rq   = &A[r][q][0];
      DATA_TYPE *restrict sum_row = sum;

      const int np_eff = _PB_NP;

      /* 1) Initialize the accumulation buffer for this (r, q). */
      for (p = 0; p < np_eff; p++)
        sum_row[p] = SCALAR_VAL(0.0);

      /* 2) Perform the matrix-vector product for this (r, q):
       *
       *    sum[p] += A[r][q][s] * C4[s][p]
       *
       * We iterate over s first and p inside, which allows:
       *   - unit-stride accesses in C4[s][p] (p contiguous),
       *   - unit-stride accesses in sum[p],
       *   - re-using A[r][q][s] across all p.
       *
       * The extra p-blocking loop (p0) keeps the inner-most loop
       * working on a small contiguous chunk of sum and C4, which can
       * be tuned with DOITGEN_TILE_P. For each fixed p, the sequence
       * of additions over s remains strictly increasing in s, exactly
       * as in the original code.
       */
      for (s = 0; s < np_eff; s++) {
        const DATA_TYPE a_rqs = A_rq[s];
        const DATA_TYPE *restrict C4_row = &C4[s][0];

        for (int p0 = 0; p0 < np_eff; p0 += DOITGEN_TILE_P) {
          const int p_end = (p0 + DOITGEN_TILE_P < np_eff) ?
                            (p0 + DOITGEN_TILE_P) : np_eff;

          /* Inner-most loop: contiguous in both sum_row and C4_row.
           * Mark as independent to encourage vectorization; there are
           * no loop-carried dependencies in p. */
#pragma GCC ivdep
          for (p = p0; p < p_end; p++) {
            sum_row[p] += a_rqs * C4_row[p];
          }
        }
      }

      /* 3) Copy the fully accumulated result back into A.
       * This is done only after all s-contributions have been
       * accumulated for every p, so A is updated in-place but
       * only after it is no longer needed for reads. */
#pragma GCC ivdep
      for (p = 0; p < np_eff; p++)
        A_rq[p] = sum_row[p];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}