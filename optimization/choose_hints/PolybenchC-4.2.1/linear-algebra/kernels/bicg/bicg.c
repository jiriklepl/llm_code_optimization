/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use local restrict-qualified pointers to tell the compiler that
     the different arrays do not alias. This improves vectorization.
   - Keep frequently used values (r[i], A[i][j], the running sum for q[i])
     in scalar variables to reduce memory traffic.
   - Use an OpenMP parallel for with an array-section reduction on s
     to exploit multi-core parallelism safely. If compiled without
     OpenMP support, the pragmas are ignored and the code remains
     correct and sequential.
   - Preserve the original loop structure and accumulation order in j
     to maintain numerically close behavior.
*/
static
void kernel_bicg[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  /* Local restrict-qualified aliases improve the compiler's ability to
     assume non-aliasing and generate better code (vectorization, LICM).
     PolyBench allocators already ensure that these arrays do not overlap. */
  DATA_TYPE (*restrict A_)[m]    = A;
  DATA_TYPE *restrict s_         = s;
  DATA_TYPE *restrict q_         = q;
  const DATA_TYPE *restrict p_   = p;
  const DATA_TYPE *restrict r_   = r;

#pragma scop
  /* Initialize s to zero.  Using SCALAR_VAL keeps the code generic for
     different DATA_TYPE choices (float, double, etc.). */
  for (i = 0; i < _PB_M; i++)
    s_[i] = SCALAR_VAL(0.0);

  /* BiCG kernel: for each row i of A:
       - q[i] = sum_j A[i][j] * p[j]
       - s[j] += r[i] * A[i][j]   for all j
     The arithmetic is equivalent to the original code, but we:
       * keep q[i] in a scalar accumulator (qi),
       * load r[i] only once per outer iteration,
       * take a row pointer Ai for A[i] to reduce address arithmetic. */

  /* The OpenMP pragma enables parallel execution across rows of A.
     - Each thread accumulates into a private copy of s_ (array-section
       reduction), and the runtime combines them at the end.
     - The reduction clause is standardized in OpenMP 4.5+ as
       reduction(+:s_[:_PB_M]).
     - When OpenMP is not enabled at compile time, this pragma is ignored
       and the code runs sequentially. */
#pragma omp parallel for private(i, j) reduction(+:s_[:_PB_M])
  for (i = 0; i < _PB_N; i++)
    {
      const DATA_TYPE ri = r_[i];
      const DATA_TYPE *restrict Ai = A_[i];
      DATA_TYPE qi = SCALAR_VAL(0.0);

      for (j = 0; j < _PB_M; j++)
	{
	  const DATA_TYPE Aij = Ai[j];

	  /* q[i] accumulation: inner product of row i with p. */
	  qi   += Aij * p_[j];

	  /* s[j] accumulation: contribution of row i to A^T * r. */
	  s_[j] += ri  * Aij;
	}

      /* Store the fully accumulated value for q[i]. This preserves the
         original accumulation order over j. */
      q_[i] = qi;
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}