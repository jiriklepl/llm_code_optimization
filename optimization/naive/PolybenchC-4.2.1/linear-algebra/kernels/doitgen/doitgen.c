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


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
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
   including the call and return. */
void kernel_doitgen[[gnu::flatten, gnu::noinline]](int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

  /* Create restrict-qualified views of the PolyBench arrays.
     PolyBench allocates A, C4 and sum as independent contiguous blocks,
     so it is safe to tell the compiler they do not alias. This improves
     vectorization and cache optimization of the inner loops. */
  DATA_TYPE (*restrict A_)[nq][np]      = A;
  const DATA_TYPE (*restrict C4_)[np]   = C4;
  DATA_TYPE *restrict sum_              = sum;

#pragma scop
  for (r = 0; r < _PB_NR; r++) {
    for (q = 0; q < _PB_NQ; q++) {

      /* Initialize the temporary accumulation buffer once per (r,q).
         In the original code, each sum[p] was initialized inside the
         outer p loop; here we perform an equivalent initialization in
         a single pass over p, which reduces work and is more cache
         friendly. */
      for (p = 0; p < _PB_NP; p++) {
        sum_[p] = SCALAR_VAL(0.0);
      }

      /* Compute:
             sum_[p] = Σ_s A_[r][q][s] * C4_[s][p]
         This is mathematically identical to the original kernel, but
         the loop nest is reorganized to improve data locality:

           - 's' is now the outer loop of the product.
             Each A_[r][q][s] is loaded once and reused across all p.
           - The inner-most loop iterates over 'p', so C4_[s][p] and
             sum_[p] are accessed with unit stride, enabling efficient
             cache use and SIMD vectorization. */
      for (s = 0; s < _PB_NP; s++) {
        DATA_TYPE a_rqs = A_[r][q][s];

        /* Tell GCC that iterations of this loop are independent,
           which, together with restrict-qualified pointers, helps it
           generate wide SIMD code. */
        #pragma GCC ivdep
        for (p = 0; p < _PB_NP; p++) {
          sum_[p] += a_rqs * C4_[s][p];
        }
      }

      /* Copy the accumulated result back into A.
         This preserves the original semantics where A[r][q][p] is
         updated only after the full dot product has been computed
         using the original A[r][q][s] values. */
      for (p = 0; p < _PB_NP; p++) {
        A_[r][q][p] = sum_[p];
      }
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