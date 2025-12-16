/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Optional: enable parallel execution of the main kernel when
 * compiled with -fopenmp. When OpenMP is not enabled, this header
 * is not included and all pragmas are ignored by the compiler. */
#ifdef _OPENMP
# include <omp.h>
#endif

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"


/* Array initialization.
 *
 * Optimizations:
 *  - Precompute 1/n once (in DATA_TYPE precision) and use
 *    multiplication instead of repeated division inside the loops.
 *  - Take row pointers for A and B to reduce address computation.
 *
 * Semantics are preserved: each element is computed from the same
 * integer expressions as in the original version.
 */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  *alpha = SCALAR_VAL(1.5);
  *beta  = SCALAR_VAL(1.2);

  /* Compute 1/n once to avoid an expensive division in the inner loop. */
  const DATA_TYPE inv_n = SCALAR_VAL(1.0) / (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)(i % n) * inv_n;

      /* Row pointers improve locality and reduce address arithmetic. */
      DATA_TYPE *Ai = A[i];
      DATA_TYPE *Bi = B[i];

      for (j = 0; j < n; j++) {
	DATA_TYPE vA = (DATA_TYPE)((i * j + 1) % n);
	DATA_TYPE vB = (DATA_TYPE)((i * j + 2) % n);

	Ai[j] = vA * inv_n;
	Bi[j] = vB * inv_n;
      }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations:
 *  - Use local restrict-qualified pointers (tmp_r, x_r, y_r) to
 *    improve alias analysis.
 *  - Use scalar accumulators (tmp_i, y_i) held in registers instead
 *    of repeatedly updating tmp[i] and y[i] in memory.
 *  - Take row pointers Ai and Bi once per outer iteration to reduce
 *    address computation in the inner loop.
 *  - Optional OpenMP parallelization over the outer loop when
 *    compiled with -fopenmp (each row i is independent).
 *
 * The arithmetic order for each inner-product is preserved
 * (sequential accumulation over j), so numerical behavior remains
 * consistent with the original code.
 */
static
void kernel_gesummv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i;
  const int nn = _PB_N;

  /* Local restrict-qualified aliases help the compiler assume that
   * these arrays do not overlap, enabling better vectorization and
   * code motion. In the PolyBench framework, the arrays are
   * separately allocated, so this assumption holds in practice. */
  DATA_TYPE *restrict tmp_r = tmp;
  DATA_TYPE *restrict x_r   = x;
  DATA_TYPE *restrict y_r   = y;

#pragma scop
  /* Parallelize across rows when OpenMP is available. Each iteration
   * of the i-loop is independent. Without -fopenmp this pragma is
   * ignored by the compiler, preserving the original serial behavior. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < nn; i++)
    {
      /* Keep partial sums in registers. */
      DATA_TYPE tmp_i = SCALAR_VAL(0.0);
      DATA_TYPE y_i   = SCALAR_VAL(0.0);

      /* Row pointers give cache-friendly, contiguous access. */
      DATA_TYPE *restrict Ai = A[i];
      DATA_TYPE *restrict Bi = B[i];

      /* Compute the two dot-products for row i:
       *   tmp_i = sum_j A[i][j] * x[j]
       *   y_i   = sum_j B[i][j] * x[j]
       */
      for (int j = 0; j < nn; j++)
        {
          DATA_TYPE xj = x_r[j];
          tmp_i += Ai[j] * xj;
          y_i   += Bi[j] * xj;
        }

      /* Write back the final values. tmp[] is preserved as in the
       * original kernel, even though it is not used later on. */
      tmp_r[i] = tmp_i;
      y_r[i]   = alpha * tmp_i + beta * y_i;
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gesummv (n, alpha, beta,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(B),
		  POLYBENCH_ARRAY(tmp),
		  POLYBENCH_ARRAY(x),
		  POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}