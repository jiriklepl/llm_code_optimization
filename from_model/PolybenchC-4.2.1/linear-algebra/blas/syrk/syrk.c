/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
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
   including the call and return.
 *
 * Optimized version:
 *   - Fuses the beta-scaling and the rank-k update so that each C[i][j]
 *     (for j <= i) is loaded and stored exactly once.
 *   - Reorders loops so that the k-dimension (inner-product) is innermost,
 *     turning the update for each (i,j) into a contiguous dot product of
 *     A[i,*] and A[j,*] in row-major storage.
 *   - Tiles the (i,j) space to keep small blocks of C and the corresponding
 *     rows of A in cache, improving data locality.
 *   - Uses a register accumulator (with manual unrolling) for the reduction
 *     over k to reduce memory traffic and expose instruction-level parallelism.
 *
 * Semantics are preserved:
 *   C[i,j] = beta * C[i,j] + alpha * sum_k A[i,k] * A[j,k],  for 0 <= j <= i < N
 *   Entries with j > i are left unchanged.
 */
static
void kernel_syrk[[gnu::flatten, gnu::noinline]](int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;
  int ii, jj;

  /* Compile-time friendly tile sizes for the (i,j) space.
   * These can be tuned for the target machine; values of 32 work well
   * for typical cache sizes on modern x86-64 CPUs. */
  const int TI = 32;
  const int TJ = 32;

  /* Use the PolyBench problem-size macros inside the SCoP. */
  const int n_local = _PB_N;
  const int m_local = _PB_M;

  const DATA_TYPE alpha_local = alpha;
  const DATA_TYPE beta_local  = beta;

//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
#pragma scop
  /* Two-level blocking over (i,j) to improve locality of C and rows of A.
   * The domain is triangular (0 <= j <= i); this is enforced inside
   * the innermost loops by capping the upper bound for j. */
  for (ii = 0; ii < n_local; ii += TI)
  {
    int i_end = ii + TI;
    if (i_end > n_local)
      i_end = n_local;

    for (jj = 0; jj < n_local; jj += TJ)
    {
      int j_end = jj + TJ;
      if (j_end > n_local)
        j_end = n_local;

      /* Process rows i in the current i-tile. */
      for (i = ii; i < i_end; ++i)
      {
        /* Pointers to the current row of A and C. */
        DATA_TYPE *Ai = A[i];
        DATA_TYPE *Ci = C[i];

        /* For this row, only columns 0..i (inclusive) belong to the
         * lower triangle. Restrict the current j-tile to that range. */
        int j_limit = i + 1;          /* logical upper bound for j on this row (exclusive) */
        if (j_limit > j_end)
          j_limit = j_end;

        /* If the current j-tile starts at or beyond j_limit, there is
         * no work for this row in this tile. */
        if (jj >= j_limit)
          continue;

        /* Inner loop over the j-dimension within the tile, limited by
         * the triangular condition j <= i. Each (i,j) pair is handled
         * exactly once over all (ii,jj) tiles. */
        for (j = jj; j < j_limit; ++j)
        {
          DATA_TYPE *Aj = A[j];

          /* Fused beta scaling: start from beta * C[i][j], then add
           * alpha times the dot product of A[i,*] and A[j,*].  */
          DATA_TYPE cij = Ci[j] * beta_local;

          /* Register reduction over k, manually unrolled by 4 to
           * increase ILP and encourage vectorization.  The accesses
           * Ai[k] and Aj[k] are both unit-stride in k (row-major A),
           * which is ideal for SIMD. */
          DATA_TYPE sum0 = 0;
          DATA_TYPE sum1 = 0;
          DATA_TYPE sum2 = 0;
          DATA_TYPE sum3 = 0;

          for (k = 0; k + 3 < m_local; k += 4)
          {
            sum0 += Ai[k]     * Aj[k];
            sum1 += Ai[k + 1] * Aj[k + 1];
            sum2 += Ai[k + 2] * Aj[k + 2];
            sum3 += Ai[k + 3] * Aj[k + 3];
          }

          DATA_TYPE sum = (sum0 + sum1) + (sum2 + sum3);

          /* Clean up any remaining iterations of k (m_local not multiple of 4). */
          for (; k < m_local; ++k)
            sum += Ai[k] * Aj[k];

          /* Final update for this (i,j): C[i][j] = beta*C[i][j] + alpha*dot(Ai,Aj). */
          Ci[j] = cij + alpha_local * sum;
        }
      }
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

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}