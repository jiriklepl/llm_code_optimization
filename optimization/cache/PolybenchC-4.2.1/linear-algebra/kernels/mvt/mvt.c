/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"

/* Tunable blocking factor for the inner j-loop of kernel_mvt.
 *
 * This controls how the j-dimension is blocked inside the main kernel.
 * It can be overridden at compile time, e.g.:
 *
 *   gcc -O3 -DNDEBUG -DMVT_BLOCK_J=128 ...
 *
 * The kernel does not rely on any specific value; the default 64 is a
 * reasonable compromise for modern cache hierarchies and vector widths.
 */
#ifndef MVT_BLOCK_J
# define MVT_BLOCK_J 64
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i]  = (DATA_TYPE) (i % n)       / n;
      x2[i]  = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;

      /* Pull the row pointer out of the inner loop to reduce address
         arithmetic cost when initializing A. */
      DATA_TYPE* Ai = A[i];

      for (j = 0; j < n; j++)
	Ai[j] = (DATA_TYPE) (i * j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt[[gnu::flatten, gnu::noinline]](int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;
  const int nPB = _PB_N;

#pragma scop
  /* Optimized kernel:
   *
   * Original code:
   *   1) x1[i] += sum_j A[i][j] * y_1[j];   (row-wise access)
   *   2) x2[i] += sum_j A[j][i] * y_2[j];   (column-wise access)
   *
   * Here we compute both x1 and x2 in a single pass over A, using only
   * row-wise accesses to A to maximize spatial locality:
   *
   *   For each matrix element A[i][j] we:
   *     - add A[i][j] * y_1[j] to x1[i]
   *     - add A[i][j] * y_2[i] to x2[j]
   *
   * That is:
   *   x1[i] = x1[i] + sum_j A[i][j] * y_1[j];
   *   x2[j] = x2[j] + sum_i A[i][j] * y_2[i];
   *
   * This is algebraically identical to the original computation of
   * x1 and x2.  Moreover, for each element x1[i] and x2[j] the
   * accumulation over the other index (j or i) is still performed in
   * strictly increasing order, so the sequence of floating-point
   * operations for each element matches the original implementation.
   *
   * We also:
   *   - keep a local accumulator (xi1) for x1[i] to reduce memory
   *     traffic and help auto-vectorization.
   *   - hoist the per-row pointer Ai = A[i] outside the inner loop to
   *     avoid repeated address calculations.
   *   - optionally block the j-dimension with MVT_BLOCK_J to allow
   *     cache and vector-width tuning.
   */
  for (i = 0; i < nPB; i++)
    {
      /* Local accumulator for x1[i]. */
      DATA_TYPE xi1 = x1[i];
      const DATA_TYPE y2_i = y_2[i];

      /* Pointer to row i of A for faster access. */
      DATA_TYPE* Ai = A[i];

      /* Block over j to allow tuning for cache / vector size.  The
         blocking does not change the order of j for any given i, so
         floating-point accumulation order is preserved. */
      for (int jj = 0; jj < nPB; jj += MVT_BLOCK_J)
        {
          const int j_end = (jj + MVT_BLOCK_J < nPB) ? (jj + MVT_BLOCK_J) : nPB;

          for (j = jj; j < j_end; j++)
            {
              const DATA_TYPE Aij = Ai[j];

              /* Contribution to x1[i] from column j (A row i, col j). */
              xi1 += Aij * y_1[j];

              /* Contribution to x2[j] from row i.
                 Original second loop: x2[i] += A[j][i] * y_2[j]
                 becomes:             x2[j] += A[i][j] * y_2[i]. */
              x2[j] += Aij * y2_i;
            }
        }

      x1[i] = xi1;
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}