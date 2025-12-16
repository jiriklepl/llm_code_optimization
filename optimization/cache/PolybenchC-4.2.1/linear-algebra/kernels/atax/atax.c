/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
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

   Optimizations applied:
   - Use local restrict-qualified pointers to improve alias analysis.
   - Keep A accessed row-wise to preserve good spatial locality.
   - Accumulate tmp[i] in a register and store it once per row.
   - Reuse the freshly computed tmp[i] value from a register for the
     y-update (avoid re-loading tmp[i] from memory).
   - Manually unroll the inner loops on j by a factor of 4 while
     preserving the original operation order (and thus the original
     floating-point reduction order).
*/
static
void kernel_atax[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  /* Use the (m, n) parameters for the array shapes to match the
     actual runtime sizes and help the compiler reason about memory. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE *restrict x_      = x;
  DATA_TYPE *restrict y_      = y;
  DATA_TYPE *restrict tmp_    = tmp;

  /* Effective loop bounds used by PolyBench. These may differ from
     the raw sizes M,N when PolyBench is configured with different
     dataset sizes or loop bounds. */
  const int mPB = _PB_M;
  const int nPB = _PB_N;

  /* Unrolling factor for the inner loops over j.
     The code below is specialized for a factor of 4, which generally
     maps well to SIMD widths on modern x86-64 CPUs while keeping the
     code simple and preserving operation order. */
  const int j_unroll      = 4;
  const int j_limit_unroll = nPB & ~(j_unroll - 1); /* largest multiple of 4 <= nPB */

#pragma scop
  /* Initialize y to zero. */
  for (j = 0; j < nPB; j++)
    y_[j] = SCALAR_VAL(0.0);

  /* Main computation:
     For each row i:
       - tmp[i] = sum_j A[i][j] * x[j];
       - y[j]  += A[i][j] * tmp[i]  for all j.
     We keep A accessed row-wise and reuse the row pointer. */
  for (i = 0; i < mPB; i++)
    {
      DATA_TYPE *restrict Ai = A_[i];

      /* Compute tmp[i] as a dot-product between row i of A and x.
         Accumulate in a scalar register to avoid repeated memory
         traffic to tmp_. */
      DATA_TYPE tmp_i = SCALAR_VAL(0.0);

      /* Manually unrolled loop on j, factor 4.
         The sequence of additions is kept identical to the original
         scalar loop (j, j+1, j+2, j+3 in order), so the floating-point
         reduction order is preserved. */
      j = 0;
      for (; j < j_limit_unroll; j += 4)
        {
          tmp_i += Ai[j]   * x_[j];
          tmp_i += Ai[j+1] * x_[j+1];
          tmp_i += Ai[j+2] * x_[j+2];
          tmp_i += Ai[j+3] * x_[j+3];
        }
      /* Handle the remaining elements (if nPB is not a multiple of 4). */
      for (; j < nPB; j++)
        tmp_i += Ai[j] * x_[j];

      /* Store the final tmp[i] value (live-out array, must be preserved). */
      tmp_[i] = tmp_i;

      /* Reuse tmp_i directly from a register for the y update to avoid
         reloading tmp_[i] from memory. */
      const DATA_TYPE ti = tmp_i;

      /* Update y using the same access pattern over A[i][j] to keep
         row-wise locality. */
      j = 0;
      for (; j < j_limit_unroll; j += 4)
        {
          y_[j]   += Ai[j]   * ti;
          y_[j+1] += Ai[j+1] * ti;
          y_[j+2] += Ai[j+2] * ti;
          y_[j+3] += Ai[j+3] * ti;
        }
      for (; j < nPB; j++)
        y_[j] += Ai[j] * ti;
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}