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
 *   - Precompute 1/m and 1/n to replace divisions inside the loops by
 *     multiplications.
 *   - Avoid redundant modulo operations: for the ranges of i used here
 *     we have (i % m) == i and (i % n) == i.
 *   - Use restrict-qualified local pointers to help alias analysis.
 */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  const DATA_TYPE inv_m = SCALAR_VAL(1.0) / (DATA_TYPE) m;
  const DATA_TYPE inv_n = SCALAR_VAL(1.0) / (DATA_TYPE) n;

  /* Local restrict-qualified aliases to help the compiler optimize. */
  DATA_TYPE (* restrict A_)[m] = A;
  DATA_TYPE * restrict r_ = r;
  DATA_TYPE * restrict p_ = p;

  /* Initialize vector p.
   * Original: p[i] = (DATA_TYPE)(i % m) / m;
   * Since 0 <= i < m, (i % m) == i.
   */
  for (i = 0; i < m; i++)
    p_[i] = (DATA_TYPE) i * inv_m;

  /* Initialize r and A.
   * Original: r[i] = (DATA_TYPE)(i % n) / n;
   *           A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
   * For r: 0 <= i < n, so (i % n) == i.
   * For A we keep the same integer pattern but reuse inv_n.
   */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE r_i = (DATA_TYPE) i * inv_n;
      r_[i] = r_i;

      for (j = 0; j < m; j++)
        {
          A_[i][j] = (DATA_TYPE) ((i * (j + 1)) % n) * inv_n;
        }
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
 * Original computation:
 *   s = A^T * r
 *   q = A   * p
 *
 * Optimizations:
 *   - Introduce restrict-qualified local pointers for better alias analysis.
 *   - Hoist r[i] out of the inner loop (use r_i).
 *   - Use a scalar accumulator q_i for q[i] to keep it in registers.
 *   - Add OpenMP pragmas:
 *       * parallel for over i with an array reduction on s
 *       * simd over the inner j-loop for better vectorization
 *     If the program is compiled without OpenMP support, these pragmas
 *     are ignored and the code executes sequentially with the same
 *     semantics as the original.
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

  /* Local restrict-qualified aliases to the arrays.  In the PolyBench
     framework these arrays are non-overlapping, so this is valid and
     allows the compiler to generate more aggressive optimizations. */
  DATA_TYPE (* restrict A_)[m] = A;
  DATA_TYPE * restrict s_ = s;
  DATA_TYPE * restrict q_ = q;
  DATA_TYPE * restrict p_ = p;
  DATA_TYPE * restrict r_ = r;

#pragma scop
  /* Initialize s exactly as in the original code. */
  for (i = 0; i < _PB_M; i++)
    s_[i] = SCALAR_VAL(0.0);

  /* Parallelize the outer i-loop.  We use an OpenMP array reduction on s_
     so that each thread accumulates into a private copy of s, which is
     combined at the end of the parallel region.  The OpenMP pragmas are
     ignored by compilers that are not invoked with OpenMP support. */
#pragma omp parallel for default(none) private(i,j) shared(A_,p_,q_,r_,s_,m,n) reduction(+:s_[:m]) schedule(static)
  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE r_i = r_[i];
      DATA_TYPE q_i = SCALAR_VAL(0.0);

      /* Vectorize the inner loop.  Each iteration updates a distinct
         element s_[j], so there is no loop-carried dependency on j
         except for the scalar accumulator q_i, which is handled via
         the reduction clause. */
#pragma omp simd reduction(+:q_i)
      for (j = 0; j < _PB_M; j++)
	{
	  DATA_TYPE a_ij = A_[i][j];
	  s_[j] += r_i * a_ij;
	  q_i   += a_ij * p_[j];
	}

      q_[i] = q_i;
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