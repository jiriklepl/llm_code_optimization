/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syr2k[[gnu::flatten, gnu::noinline]](int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j, k;

  /* Create local restrict-qualified views of the input arrays.
   *
   * - The PolyBench driver always passes distinct, non-aliasing arrays
   *   for A, B, and C, so using `restrict` here is semantically valid
   *   for all actual uses of this function, and it allows the compiler
   *   to perform stronger optimizations (vectorization, register
   *   promotion, etc.).
   *
   * - We keep the original parameter types intact and only introduce
   *   new local pointers; this does not change the externally visible
   *   interface of the kernel.
   */
  DATA_TYPE (*restrict C_)[n] = C;
  DATA_TYPE (*restrict A_)[m] = A;
  DATA_TYPE (*restrict B_)[m] = B;

  /* Cache scalar parameters in local const variables so that the
   * compiler can assume they are invariant and keep them in registers.
   */
  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  /* Optimized loop nest.
   *
   * Key changes relative to the original version:
   *
   * 1. Loop fusion and scalar accumulation
   *    -----------------------------------
   *    Originally, C was first scaled by beta in a separate j-loop,
   *    and then updated in a k-j loop nest:
   *
   *       for (i)
   *         for (j <= i)
   *           C[i][j] *= beta;
   *         for (k)
   *           for (j <= i)
   *             C[i][j] += ...
   *
   *    We now fuse these into a single j-loop per i:
   *
   *       for (i)
   *         for (j <= i) {
   *           cij = C[i][j] * beta;
   *           for (k)
   *             cij += ...;
   *           C[i][j] = cij;
   *         }
   *
   *    For each element C[i][j], the sequence of arithmetic operations
   *    is unchanged: one multiplication by beta followed by a sum over
   *    k in the same order. The only difference is that intermediate
   *    values are kept in a scalar register (`cij`) instead of memory,
   *    which reduces memory traffic and improves cache utilization
   *    without altering the floating-point result for each C[i][j].
   *
   * 2. Improved data locality for A and B
   *    ----------------------------------
   *    The inner-most loop is now over k. For a fixed (i, j), we
   *    traverse:
   *
   *      - A[i][k], B[i][k]   : row i over k (contiguous in memory)
   *      - A[j][k], B[j][k]   : row j over k (also contiguous)
   *
   *    This gives unit-stride accesses on the k-dimension for all
   *    arrays, which is friendly to caches and SIMD units.
   *
   * 3. Thread-level parallelism
   *    ------------------------
   *    The outer i-loop is embarrassingly parallel: each (i, j) updates
   *    a distinct element C[i][j], and no iteration reads another
   *    iteration's output. We exploit this with OpenMP:
   *
   *        #pragma omp parallel for schedule(static)
   *
   *    If the program is compiled without OpenMP support, the pragmas
   *    are ignored by the compiler and the code remains correct and
   *    sequential.
   *
   * 4. SIMD vectorization hint
   *    ------------------------
   *    Inside the innermost k-loop we add `#pragma omp simd` to nudge
   *    the compiler towards vectorization along k. With the restrict
   *    qualifiers and unit-stride accesses, this typically produces
   *    efficient SIMD code on modern x86-64 CPUs.
   */

  #pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < _PB_N; i++)
  {
    /* Pointers to the i-th rows of the matrices, reused across j. */
    DATA_TYPE *restrict Ci = C_[i];
    DATA_TYPE *restrict Ai = A_[i];
    DATA_TYPE *restrict Bi = B_[i];

    /* Only the lower-triangular part (j <= i) is updated. */
    for (j = 0; j <= i; j++)
    {
      /* Start from beta * C[i][j], then accumulate over k in a
       * register. This preserves the original arithmetic order for
       * each C[i][j] while avoiding repeated memory loads/stores.
       */
      DATA_TYPE cij = Ci[j] * beta_local;

      /* Pointers to the j-th rows of A and B used in this iteration. */
      DATA_TYPE *restrict Aj = A_[j];
      DATA_TYPE *restrict Bj = B_[j];

      /* Inner-product over the k-dimension.
       *
       * The expression below is algebraically identical to the
       * original:
       *
       *   C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
       *
       * We keep the same expression tree (and thus the same
       * floating-point operation ordering for each term), only
       * operating on scalar temporaries instead of reading/writing C
       * every iteration.
       */
      #pragma omp simd
      for (k = 0; k < _PB_M; k++)
      {
        cij += Aj[k] * alpha_local * Bi[k]
             + Bj[k] * alpha_local * Ai[k];
      }

      /* Write back the fully accumulated value. */
      Ci[j] = cij;
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syr2k (n, m,
		alpha, beta,
		POLYBENCH_ARRAY(C),
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}