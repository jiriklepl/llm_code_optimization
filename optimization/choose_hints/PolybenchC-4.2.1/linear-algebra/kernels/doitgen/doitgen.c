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

  /* Create local restrict-qualified aliases to help the compiler
   * with alias analysis and vectorization.  The shapes use the
   * runtime sizes (nr, nq, np) provided to the kernel. */
  DATA_TYPE (*restrict Aref)[nq][np] = A;
  DATA_TYPE (*restrict C4ref)[np]    = C4;
  DATA_TYPE *restrict sumref         = sum;

#pragma scop
  /* The original kernel computes, for each (r,q):
   *
   *   for p:
   *     sum[p] = 0;
   *     for s:
   *       sum[p] += A[r][q][s] * C4[s][p];
   *   for p:
   *     A[r][q][p] = sum[p];
   *
   * This is a matrix–vector product:
   *   new_row[p] = Σ_s A[r][q][s] * C4[s][p].
   *
   * Optimizations applied:
   *  - Parallelize the (r,q) space (each (r,q) is independent).
   *  - Use a private accumulation buffer per (r,q) to enable parallelism
   *    and keep data in L1 cache.
   *  - Reorder the inner loops so that the innermost loop runs over `p`,
   *    giving unit-stride access to C4[s][p] (row-major layout) and
   *    enabling efficient vectorization.
   */

#pragma omp parallel for collapse(2) private(p,s) schedule(static)
  for (r = 0; r < _PB_NR; r++)
  {
    for (q = 0; q < _PB_NQ; q++)
    {
      /* Per-(r,q) accumulation buffer.
       * As a VLA on the stack, this is naturally private to each
       * OpenMP iteration (and each thread when OpenMP is enabled). */
      DATA_TYPE sum_local[_PB_NP];

      /* Initialize all components of the accumulation buffer to 0.0.
       * In the original code, each sum[p] was initialized inside the
       * p-loop. Doing it once per (r,q) is equivalent and reduces work. */
      for (p = 0; p < _PB_NP; p++)
        sum_local[p] = SCALAR_VAL(0.0);

      /* Reordered accumulation:
       *
       *   for s:
       *     a = A[r][q][s];
       *     for p:
       *       sum_local[p] += a * C4[s][p];
       *
       * For each fixed p, the sequence of updates across s is
       * identical to the original kernel; only operations for
       * different p's are interleaved, which does not change the
       * result of each independent reduction. */
      for (s = 0; s < _PB_NP; s++)
      {
        DATA_TYPE a_rqs = Aref[r][q][s];
        DATA_TYPE *restrict c4_row = &C4ref[s][0];

        /* Hint to GCC that there are no loop-carried dependencies
         * across iterations of this inner loop, helping vectorization. */
#pragma GCC ivdep
        for (p = 0; p < _PB_NP; p++)
        {
          sum_local[p] += a_rqs * c4_row[p];
        }
      }

      /* Write back the computed row into A.
       * This matches the original assignment A[r][q][p] = sum[p]. */
      for (p = 0; p < _PB_NP; p++)
        Aref[r][q][p] = sum_local[p];
    }
  }
#pragma endscop

  /* Restore the final state of the `sum` array so that it matches
   * the original kernel: after all loops, `sum[p]` equals
   * A[NR-1][NQ-1][p].  This has no impact on the benchmark's
   * correctness checking (which only uses A), but preserves the
   * observable final state of `sum`. */
  if (_PB_NR > 0 && _PB_NQ > 0)
  {
    int r_last = _PB_NR - 1;
    int q_last = _PB_NQ - 1;
    for (p = 0; p < _PB_NP; p++)
      sumref[p] = Aref[r_last][q_last][p];
  }
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