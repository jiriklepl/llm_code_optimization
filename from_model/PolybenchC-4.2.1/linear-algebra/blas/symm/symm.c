/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  /* Local copies to help the compiler keep scalars in registers. */
  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

  /* Use explicit restrict-qualified pointers for better alias analysis.
   * In the PolyBench setup, A, B, and C are distinct arrays.
   */
  DATA_TYPE (* restrict C_)[n] = C;
  DATA_TYPE (* restrict A_)[m] = A;
  DATA_TYPE (* restrict B_)[n] = B;

  int i, j, k;

  /* Tile size along the column (j) dimension.
   * This improves cache locality and gives the compiler a clear
   * unit-stride innermost loop to vectorize.
   * A value in the range 32–256 is typically reasonable; 64 is a
   * conservative, cache-friendly choice.
   */
  const int tile_j = 64;

  /* Temporary buffers over columns (reused for each row i).
   * Size n is sufficient since _PB_N <= n (PolyBench convention).
   * These are stack-allocated VLAs; they do not change asymptotic
   * memory usage (O(n) extra).
   */
  DATA_TYPE temp1[n];
  DATA_TYPE temp2[n];

  //BLAS PARAMS
  //SIDE = 'L'
  //UPLO = 'L'
  // =>  Form  C := alpha*A*B + beta*C
  // A is MxM (only lower triangle used, representing a symmetric matrix)
  // B is MxN
  // C is MxN
  //note that due to Fortran array layout, the original code resembles
  //the upper triangular case in BLAS, but mathematically it computes
  //C := alpha * Symm(A_lower) * B + beta * C.

#pragma scop
  /* 1. Scale C by beta in a separate pass.
   *
   * Original kernel multiplies each C[i][j] by beta exactly once
   * (when processing row i). Doing it here up front is algebraically
   * equivalent:
   *
   *   C_final = beta * C0 + alpha * (symmetric product)
   *
   * This pass is simple, fully vectorizable, and touches C in unit
   * stride (good cache and SIMD behavior).
   */
  for (i = 0; i < _PB_M; i++) {
    for (j = 0; j < _PB_N; j++) {
      C_[i][j] *= beta_local;
    }
  }

  /* 2. Symmetric matrix-matrix multiply using only the stored
   *    lower triangle of A, with column blocking and 1D temporaries.
   *
   * For each row i:
   *   - temp1[j] = alpha * B[i, j]         (pre-scaled row of B)
   *   - temp2[j] = 0
   *   - For each k < i:
   *       * C[k, j] += A[i, k] * temp1[j]  (contribution from S(k,i))
   *       * temp2[j] += A[i, k] * B[k, j]  (accumulate contribution to row i)
   *   - Then for the diagonal and accumulated lower part:
   *       C[i, j] += A[i, i] * temp1[j] + alpha * temp2[j]
   *
   * Together with the initial C *= beta pass, this is exactly the
   * same computation as the original kernel (up to harmless
   * reordering of additions in floating point):
   *
   *   C[i, j] = beta * C0[i, j]
   *             + alpha * (
   *                   A[i, i] * B[i, j]
   *                 + sum_{k <  i} A[i, k] * B[k, j]
   *                 + sum_{k >  i} A[k, i] * B[k, j]
   *               )
   *
   * The k > i contributions to row i are produced when processing
   * larger outer indices (i' > i) and updating C[i, j] via C[k][j]
   * in the loop below, just as in the original PolyBench kernel.
   *
   * The key optimization is that the innermost loops now run over j,
   * which is contiguous in memory for B and C, enabling efficient
   * SIMD vectorization and better cache utilization.
   */
  for (i = 0; i < _PB_M; i++) {

    /* Process C and B in column tiles [J, j_end). */
    for (int J = 0; J < _PB_N; J += tile_j) {
      int j_end = J + tile_j;
      if (j_end > _PB_N)
        j_end = _PB_N;
      int jb_len = j_end - J;

      /* Precompute temp1 (alpha * B[i, j]) and zero temp2
       * for this column block.
       */
      for (j = 0; j < jb_len; ++j) {
        int jj = J + j;
        DATA_TYPE bij = B_[i][jj];
        temp1[j] = alpha_local * bij;
        temp2[j] = (DATA_TYPE)0;
      }

      /* Handle strictly lower part: k = 0 .. i-1.
       * For each (i, k) with k < i we:
       *   - update row k of C (C[k, J:j_end])
       *   - accumulate partial sums into temp2 for row i
       *
       * Only A[i, k] with k <= i are ever read, so we stay
       * strictly within the initialized lower triangle of A.
       */
      for (k = 0; k < i; ++k) {
        const DATA_TYPE a_ik = A_[i][k];

        /* Pointers to the current row segments for B and C. */
        DATA_TYPE * restrict Bk = &B_[k][J];
        DATA_TYPE * restrict Ck = &C_[k][J];

        for (j = 0; j < jb_len; ++j) {
          Ck[j]      += a_ik * temp1[j];   /* C[k, j] += alpha * A[i,k] * B[i,j] */
          temp2[j]   += a_ik * Bk[j];      /* accumulate A[i,k] * B[k,j]        */
        }
      }

      /* Diagonal and accumulated lower-triangular contributions for row i. */
      {
        const DATA_TYPE a_ii = A_[i][i];
        DATA_TYPE * restrict Ci = &C_[i][J];

        for (j = 0; j < jb_len; ++j) {
          /* temp1[j] = alpha * B[i, j],
           * temp2[j] = sum_{k<i} A[i,k] * B[k,j]
           *
           * So:
           *   a_ii * temp1[j] = alpha * A[i,i] * B[i,j]
           *   alpha * temp2[j] = alpha * sum_{k<i} A[i,k] * B[k,j]
           */
          Ci[j] += a_ii * temp1[j] + alpha_local * temp2[j];
        }
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}