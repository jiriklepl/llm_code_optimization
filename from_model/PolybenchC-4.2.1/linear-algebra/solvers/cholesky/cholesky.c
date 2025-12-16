/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use an auxiliary 1D array inv_diag[i] that stores 1 / A[i][i] after
     each diagonal is computed. This replaces O(N^2) divisions in the
     off-diagonal updates with cheaper multiplications, leaving only O(N)
     divisions (one per row).
   - Promote rows of A into local pointers (A_row_i, A_row_j) annotated
     with restrict to help the compiler with alias analysis.
   - Tile the outer i-loop to improve cache locality when working on
     consecutive rows.
   - Rewrite the inner products (over k) as explicit dot products
     accumulated into scalar temporaries and manually unroll them by a
     factor of 4, which improves instruction-level parallelism and
     exposes a vectorizable pattern to the compiler.
   - Preserve the original left-looking algorithm and loop-carried
     dependencies: overall numerical behavior remains that of the
     original Cholesky factorization.
*/
static
void kernel_cholesky[[gnu::flatten, gnu::noinline]](int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;
  int I;

  /* _PB_N is the possibly "reduced" problem size used by PolyBench for
     performance evaluation. The physical leading dimension of A is n. */
  const int n_full = n;
  const int n_iter = _PB_N;

  /* Add a restrict-qualified alias for A to help the compiler with
     alias analysis and vectorization. A is an n_full x n_full
     row-major matrix. */
  DATA_TYPE (*restrict A_)[n_full] = A;

  /* Temporary 1D array to store reciprocals of diagonal entries:
       inv_diag[i] = 1.0 / A[i][i]  (after A[i][i] is finalized).
     Size is O(N) and thus negligible compared to the O(N^2) matrix. */
  DATA_TYPE inv_diag[n_iter];

  /* Tile size for the outer i loop. This is a trade-off between keeping
     a block of rows in cache and loop overhead. 32 is a reasonable
     default for typical PolyBench sizes on modern x64 CPUs. */
  const int tile_i = 32;

#pragma scop
  /* Process the matrix row by row, but in tiles along the i dimension
     to improve temporal locality. Tiling does not change the global
     order of i: rows are still processed in strictly increasing order. */
  for (I = 0; I < n_iter; I += tile_i)
  {
    int i_end = I + tile_i;
    if (i_end > n_iter)
      i_end = n_iter;

    for (i = I; i < i_end; ++i)
    {
      DATA_TYPE *restrict A_row_i = A_[i];

      /* ---------------- Off-diagonal entries: j = 0..i-1 ----------------
         Original code:

           for (j = 0; j < i; j++) {
             for (k = 0; k < j; k++)
               A[i][j] -= A[i][k] * A[j][k];
             A[i][j] /= A[j][j];
           }

         We rewrite this as:

           dot_ij = sum_{k=0}^{j-1} A[i][k] * A[j][k]
           A[i][j] = (A[i][j] - dot_ij) * inv_diag[j];

         where inv_diag[j] = 1 / A[j][j], computed when row j was
         finalized. This is algebraically equivalent but uses only a
         single division per row instead of per (i,j) pair.
      */
      for (j = 0; j < i; ++j)
      {
        DATA_TYPE *restrict A_row_j = A_[j];
        DATA_TYPE sum = (DATA_TYPE)0;

        /* Dot product: sum_{k=0}^{j-1} A[i][k] * A[j][k]
           Manually unrolled by a factor of 4 to expose ILP and
           facilitate SIMD vectorization by the compiler. */
        int k_limit  = j;
        int k_unroll = k_limit & ~3;  /* largest multiple of 4 <= j */

        for (k = 0; k < k_unroll; k += 4)
        {
          DATA_TYPE a0 = A_row_i[k];
          DATA_TYPE b0 = A_row_j[k];
          DATA_TYPE a1 = A_row_i[k + 1];
          DATA_TYPE b1 = A_row_j[k + 1];
          DATA_TYPE a2 = A_row_i[k + 2];
          DATA_TYPE b2 = A_row_j[k + 2];
          DATA_TYPE a3 = A_row_i[k + 3];
          DATA_TYPE b3 = A_row_j[k + 3];

          sum += a0 * b0
               + a1 * b1
               + a2 * b2
               + a3 * b3;
        }

        /* Remainder loop for k values not covered by the unrolled loop. */
        for (; k < k_limit; ++k)
        {
          sum += A_row_i[k] * A_row_j[k];
        }

        /* Use precomputed reciprocal of the diagonal entry A[j][j]. */
        A_row_i[j] = (A_row_i[j] - sum) * inv_diag[j];
      }

      /* ---------------- Diagonal entry: j == i ----------------
         Original code:

           for (k = 0; k < i; k++)
             A[i][i] -= A[i][k] * A[i][k];
           A[i][i] = SQRT_FUN(A[i][i]);

         We rewrite this using an explicit reduction:

           sum_ii = sum_{k=0}^{i-1} A[i][k]^2
           A[i][i] = SQRT_FUN(A[i][i] - sum_ii);

         which is mathematically equivalent and improves the chances of
         efficient vectorization of the inner loop.
      */
      DATA_TYPE sum_diag = (DATA_TYPE)0;
      int k_limit2  = i;
      int k_unroll2 = k_limit2 & ~3;  /* largest multiple of 4 <= i */

      for (k = 0; k < k_unroll2; k += 4)
      {
        DATA_TYPE v0 = A_row_i[k];
        DATA_TYPE v1 = A_row_i[k + 1];
        DATA_TYPE v2 = A_row_i[k + 2];
        DATA_TYPE v3 = A_row_i[k + 3];

        sum_diag += v0 * v0
                  + v1 * v1
                  + v2 * v2
                  + v3 * v3;
      }

      for (; k < k_limit2; ++k)
      {
        DATA_TYPE v = A_row_i[k];
        sum_diag += v * v;
      }

      A_row_i[i] = SQRT_FUN(A_row_i[i] - sum_diag);

      /* Precompute the reciprocal of the diagonal for use in
         subsequent rows' off-diagonal updates. This replaces
         divisions by multiplications there. */
      inv_diag[i] = (DATA_TYPE)1.0 / A_row_i[i];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}