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

/* Enable OpenMP-based parallelism when compiled with -fopenmp. */
#ifdef _OPENMP
#  include <omp.h>
#endif

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

  /* Initialize A with unit-stride access in the innermost dimension
     to respect the row-major layout and maximize cache friendliness. */
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;

  /* Initialize C4 similarly (row-major, inner loop over last dimension). */
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

   Original (mathematically):

     for each (r,q):
       for each p:
         sum[p] = 0;
         for each s:
           sum[p] += A[r][q][s] * C4[s][p];
       A[r][q][p] = sum[p];

   This computes, for each fixed (r,q), a matrix-vector product:
   A[r][q][*] <- C4^T * A[r][q][*].

   Optimizations applied:
   - Loop reordering inside the (p,s) nest:
       * We first zero the entire sum vector.
       * Then, for each s, we load A[r][q][s] once and update all p:
           sum[p] += A[r][q][s] * C4[s][p]
         This keeps the access to C4 row-contiguous (inner loop over p),
         improving cache and vectorization, while preserving the exact
         arithmetic order for each p.
   - OpenMP parallelization across (r,q) to exploit multi-core CPUs.
   - When OpenMP is enabled, each thread uses a private accumulator
     buffer to avoid races. To preserve the observable behavior of the
     original scalar code (final contents of `sum`), the last (r,q)
     pair is computed sequentially using the original `sum` array.
*/
void kernel_doitgen[[gnu::flatten, gnu::noinline]](int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop

#if defined(_OPENMP)

  /* Parallel version: all (r,q) pairs except the last one. */
  const int last_r = _PB_NR - 1;
  const int last_q = _PB_NQ - 1;

  /* Parallelize across the outer (r,q) loops. Each thread gets a
     private accumulator vector "tmp" to avoid sharing `sum`.          */
#pragma omp parallel private(r, q, p, s)
  {
    /* One private accumulation buffer per thread.
       Size is _PB_NP (runtime), allocated as a VLA on the stack.      */
    DATA_TYPE tmp[_PB_NP];

#pragma omp for collapse(2) schedule(static)
    for (r = 0; r < _PB_NR; r++) {
      for (q = 0; q < _PB_NQ; q++) {

        /* Skip the very last (r,q) pair; it will be computed
           sequentially below using the original `sum` buffer so that
           its final value matches the scalar reference exactly.       */
        if (r == last_r && q == last_q)
          continue;

        /* Initialize local accumulator vector. */
        for (p = 0; p < _PB_NP; p++)
          tmp[p] = SCALAR_VAL(0.0);

        /* Cache-friendly and vectorizable matrix-vector product:
           - Outer loop over s
           - Inner loop over p (unit stride in C4 row)                 */
        for (s = 0; s < _PB_NP; s++) {
          const DATA_TYPE a_rqs = A[r][q][s];
          const DATA_TYPE * restrict c4_row = &C4[s][0];

          for (p = 0; p < _PB_NP; p++) {
            tmp[p] += a_rqs * c4_row[p];
          }
        }

        /* Store result back into A. */
        for (p = 0; p < _PB_NP; p++)
          A[r][q][p] = tmp[p];
      }
    }
  } /* end parallel region */

  /* Compute the last (r,q) pair sequentially using the original
     `sum` workspace, reproducing the scalar code's side effects
     exactly (including the final content of `sum`).                  */
  if (_PB_NR > 0 && _PB_NQ > 0) {
    const int r_last = last_r;
    const int q_last = last_q;

    for (p = 0; p < _PB_NP; p++)
      sum[p] = SCALAR_VAL(0.0);

    for (s = 0; s < _PB_NP; s++) {
      const DATA_TYPE a_rqs = A[r_last][q_last][s];
      const DATA_TYPE * restrict c4_row = &C4[s][0];

      for (p = 0; p < _PB_NP; p++)
        sum[p] += a_rqs * c4_row[p];
    }

    for (p = 0; p < _PB_NP; p++)
      A[r_last][q_last][p] = sum[p];
  }

#else  /* !_OPENMP : optimized sequential version */

  /* Sequential version:
     - Uses the original `sum` array as workspace.
     - Applies the same loop reordering for better data locality and
       vectorization, while preserving arithmetic order per p.        */
  for (r = 0; r < _PB_NR; r++) {
    for (q = 0; q < _PB_NQ; q++) {

      /* Initialize the accumulator vector. */
      for (p = 0; p < _PB_NP; p++)
        sum[p] = SCALAR_VAL(0.0);

      /* Matrix-vector product with inner unit-stride loop over p. */
      for (s = 0; s < _PB_NP; s++) {
        const DATA_TYPE a_rqs = A[r][q][s];
        const DATA_TYPE * restrict c4_row = &C4[s][0];

        for (p = 0; p < _PB_NP; p++)
          sum[p] += a_rqs * c4_row[p];
      }

      /* Write back into A. */
      for (p = 0; p < _PB_NP; p++)
        A[r][q][p] = sum[p];
    }
  }

#endif /* _OPENMP */

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