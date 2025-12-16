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

   Original mathematical kernel:
     tmp[i] = sum_j A[i,j] * x[j]      (tmp = A * x)
     y[j]   = sum_i A[i,j] * tmp[i]    (y   = A^T * tmp)

   This optimized version keeps exactly the same computation but
   restructures the loops to improve:
     - data locality (especially for y and rows of A),
     - opportunities for vectorization,
     - register usage (scalar tmp accumulator),
     - and control overhead (simple manual unrolling).

   Semantics are preserved exactly:
     * y is still initialized to 0;
     * tmp[i] is computed once per row with the same summation order;
     * each y[j] receives contributions from rows i in the same i-order
       as in the original code.
*/
static
void kernel_atax[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  /* Create local restrict-qualified views of the arrays to help the
     compiler with alias analysis and vectorization.  This does not
     change behavior for this benchmark, where the arrays do not
     overlap in memory. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE *restrict x_      = x;
  DATA_TYPE *restrict y_      = y;
  DATA_TYPE *restrict tmp_    = tmp;

#pragma scop
  /* 1. Initialize y to zero (same as the first loop in the original kernel). */
  for (j = 0; j < _PB_N; j++)
    y_[j] = SCALAR_VAL(0.0);

  /* 2. Compute tmp = A * x.
        We keep the per-row accumulator in a scalar register and
        perform a simple unrolled dot product to expose ILP and make
        vectorization easier for the compiler. */
  for (i = 0; i < _PB_M; i++)
    {
      DATA_TYPE acc = SCALAR_VAL(0.0);
      DATA_TYPE *restrict Ai = A_[i];

      /* Manual unrolling by 4 with a clean remainder loop.
         This preserves the exact summation order j = 0,1,2,...,N-1. */
      for (j = 0; j + 3 < _PB_N; j += 4)
      {
        acc += Ai[j    ] * x_[j    ];
        acc += Ai[j + 1] * x_[j + 1];
        acc += Ai[j + 2] * x_[j + 2];
        acc += Ai[j + 3] * x_[j + 3];
      }
      for (; j < _PB_N; j++)
        acc += Ai[j] * x_[j];

      tmp_[i] = acc;
    }

  /* 3. Compute y = A^T * tmp.
        We traverse y in tiles along j (columns) so that a small slice
        of y (and the corresponding columns of A) remains in cache
        while accumulating contributions from all rows i.

        The order of updates to each y[j] with respect to the row
        index i is unchanged: for a fixed j, we still apply updates
        for i = 0,1,...,_PB_M-1 in that order, exactly as in the
        original nested loop. */
  const int TILE_J = 256; /* tile size along j; must be > 0 */

  for (int jj = 0; jj < _PB_N; jj += TILE_J)
    {
      int j_end = jj + TILE_J;
      if (j_end > _PB_N)
        j_end = _PB_N;

      for (i = 0; i < _PB_M; i++)
        {
          DATA_TYPE t = tmp_[i];
          DATA_TYPE *restrict Ai = A_[i];

          /* As above, manually unroll the inner j-loop by 4 within
             the current tile.  This loop only updates distinct
             entries of y_, so iterations are independent. */
          int j_local;
          for (j_local = jj; j_local + 3 < j_end; j_local += 4)
          {
            y_[j_local    ] += Ai[j_local    ] * t;
            y_[j_local + 1] += Ai[j_local + 1] * t;
            y_[j_local + 2] += Ai[j_local + 2] * t;
            y_[j_local + 3] += Ai[j_local + 3] * t;
          }
          for (; j_local < j_end; j_local++)
            y_[j_local] += Ai[j_local] * t;
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