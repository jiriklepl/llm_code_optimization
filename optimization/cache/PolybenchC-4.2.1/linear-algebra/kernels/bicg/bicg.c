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


/* Array initialization.
 *
 * Optimizations:
 *  - Use local restrict-qualified pointers to help the compiler
 *    assume non-aliasing between A, r, and p.
 *  - Use a row pointer for A to improve generated code quality.
 */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  /* Local aliases with restrict to aid optimization.
     We do not access the original names (A, r, p) after this,
     so the restrict assumptions are valid. */
  DATA_TYPE (*restrict A_)[M] = (DATA_TYPE (*restrict)[M])A;
  DATA_TYPE *restrict r_ = (DATA_TYPE *restrict)r;
  DATA_TYPE *restrict p_ = (DATA_TYPE *restrict)p;

  for (i = 0; i < m; i++)
    p_[i] = (DATA_TYPE)(i % m) / m;

  for (i = 0; i < n; i++) {
    r_[i] = (DATA_TYPE)(i % n) / n;
    DATA_TYPE *restrict Ai = A_[i];
    for (j = 0; j < m; j++)
      Ai[j] = (DATA_TYPE) (i*(j+1) % n) / n;
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
 *
 * Optimizations:
 *  - Introduce local restrict-qualified aliases for all arrays to
 *    enable better auto-vectorization and eliminate assumed aliasing.
 *  - Cache A's row pointer and r[i] in registers per outer iteration.
 *  - Accumulate q[i] in a scalar register (qi) and write it back
 *    once per i instead of updating q[i] inside the inner loop.
 *    This eliminates repeated loads/stores of q[i] and keeps its
 *    arithmetic order identical to the original code.
 *  - Keep loop structure and bounds affine to preserve the SCoP
 *    (polyhedral) region marked by #pragma scop/endscop.
 */
static
void kernel_bicg [[gnu::flatten, gnu::noinline]] (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  /* Local restrict-qualified aliases.
     After these declarations we only use the underscored names,
     so the non-aliasing guarantees are respected. */
  DATA_TYPE (*restrict A_)[M]       = (DATA_TYPE (*restrict)[M])A;
  DATA_TYPE *restrict s_            = (DATA_TYPE *restrict)s;
  DATA_TYPE *restrict q_            = (DATA_TYPE *restrict)q;
  const DATA_TYPE *restrict p_      = (const DATA_TYPE *restrict)p;
  const DATA_TYPE *restrict r_      = (const DATA_TYPE *restrict)r;

  /* Cache loop bounds in local consts; helps some compilers avoid
     re-reading macros and enables potential further tuning. */
  const int m_bound = _PB_M;
  const int n_bound = _PB_N;

#pragma scop
  /* Initialize s to zero. */
  for (i = 0; i < m_bound; i++)
    s_[i] = SCALAR_VAL(0.0);

  /* BiCG kernel:
     Original operations:
       s[j] += r[i] * A[i][j];
       q[i] += A[i][j] * p[j];

     We keep the same iteration order (i outer, j inner) and
     arithmetic order over j, but:
       - r[i] is loaded once per i,
       - A[i][j] is loaded once per (i,j) and reused twice,
       - q[i] is accumulated in a register and stored once. */
  for (i = 0; i < n_bound; i++)
    {
      const DATA_TYPE ri = r_[i];
      const DATA_TYPE *restrict Ai = A_[i]; /* Row pointer for A[i][*] */
      DATA_TYPE qi = SCALAR_VAL(0.0);

      for (j = 0; j < m_bound; j++)
	{
	  const DATA_TYPE Aij = Ai[j];

	  /* q[i] = q[i] + A[i][j] * p[j]; */
	  qi += Aij * p_[j];

	  /* s[j] = s[j] + r[i] * A[i][j]; */
	  s_[j] += ri * Aij;
	}

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