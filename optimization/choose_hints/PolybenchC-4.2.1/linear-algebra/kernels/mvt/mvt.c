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
#include <stdlib.h>  /* added for calloc/free used in the optimized kernel */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"


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
      /* The (i % n) is redundant because i < n, but we keep the
         original expressions to preserve exact initialization
         semantics. */
      x1[i]  = (DATA_TYPE) (i % n)       / n;
      x2[i]  = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i * j % n) / n;
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
   including the call and return.

   Optimizations applied:
   ----------------------
   1. Use local 'restrict' pointers to help the compiler with
      alias analysis and vectorization.
   2. Keep x1[i] in a scalar register while computing A * y_1,
      reducing memory traffic for x1.
   3. Reorder the loops of the second matrix-vector product so that
      A is accessed row-wise (unit stride), improving cache locality.
      For each x2[i], the sequence of additions over j is preserved,
      so the sequential path is bitwise identical to the original.
   4. Optional OpenMP parallelization (enabled only when compiled
      with -fopenmp, i.e., when _OPENMP is defined):
        - First kernel: parallel over rows i (no dependencies).
        - Second kernel: use an auxiliary buffer x2_tmp and an
          OpenMP array reduction to accumulate A^T * y_2 in parallel,
          then add it to the original x2. This avoids data races.
      Without -fopenmp, all OpenMP-specific code is excluded at
      compile time and the behavior is purely sequential.
*/
static
void kernel_mvt[[gnu::flatten, gnu::noinline]](int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  /* Local 'restrict' views of the data.  These are safe because the
     PolyBench harness always passes distinct, non-overlapping arrays. */
  const DATA_TYPE (* restrict A_)[n] = (const DATA_TYPE (* restrict)[n])A;
  DATA_TYPE * restrict x1_           = x1;
  DATA_TYPE * restrict x2_           = x2;
  const DATA_TYPE * restrict y1_     = y_1;
  const DATA_TYPE * restrict y2_     = y_2;

  const int nPB = _PB_N;

#ifdef _OPENMP
  /* Temporary buffer used only in the OpenMP path for a thread-safe
     accumulation of A^T * y_2.  It is initialized to zero so that
     the OpenMP reduction produces exactly the pure matrix-vector
     product, which we then add to the original x2.
     Size is O(N), which is negligible compared to A (O(N^2)). */
  DATA_TYPE * restrict x2_tmp =
    (DATA_TYPE*)calloc((size_t)nPB, sizeof(DATA_TYPE));
#endif

#pragma scop

#ifdef _OPENMP
  if (x2_tmp)
  {
    /* OpenMP-parallel version.
       We use a single parallel region to amortize thread-startup
       overhead across both kernels. */
#pragma omp parallel
    {
      int ii, jj;

      /* First matrix-vector product:
           x1 = x1 + A * y_1
         Each iteration over ii updates a distinct x1[ii], so we
         can safely parallelize the outer loop. */
#pragma omp for schedule(static)
      for (ii = 0; ii < nPB; ++ii)
      {
        DATA_TYPE xi = x1_[ii];
        const DATA_TYPE * restrict Ai = A_[ii];

        for (jj = 0; jj < nPB; ++jj)
          xi += Ai[jj] * y1_[jj];

        x1_[ii] = xi;
      }

      /* Second matrix-vector product:
           x2 = x2 + A^T * y_2

         To improve cache locality, we traverse A row-wise (outer
         loop over rows jj).  Because all iterations update the
         *same* vector, we use an OpenMP array reduction on x2_tmp
         to avoid data races.  Each thread accumulates into its own
         private copy of x2_tmp[0:nPB], and at the end these are
         summed into the shared x2_tmp buffer.
       */
#pragma omp for schedule(static) reduction(+:x2_tmp[0:nPB])
      for (jj = 0; jj < nPB; ++jj)
      {
        const DATA_TYPE y2j = y2_[jj];
        const DATA_TYPE * restrict Aj = A_[jj];

        for (ii = 0; ii < nPB; ++ii)
          x2_tmp[ii] += Aj[ii] * y2j;
      }
    } /* end of parallel region */

    /* Combine the pre-existing contents of x2 with the
       newly computed A^T * y_2 stored in x2_tmp. */
    for (i = 0; i < nPB; ++i)
      x2_[i] += x2_tmp[i];
  }
  else
#endif /* _OPENMP */
  {
    /* Purely sequential version (also used as a fallback if
       allocation of x2_tmp fails in the OpenMP build). */

    /* First matrix-vector product:
         x1 = x1 + A * y_1

       Optimization: keep x1[i] in a scalar register while walking
       along row i of A.  This reduces the number of loads/stores
       of x1[i] from O(N^2) to O(N).
       The order of additions over j is unchanged. */
    for (i = 0; i < nPB; ++i)
    {
      DATA_TYPE xi = x1_[i];
      const DATA_TYPE * restrict Ai = A_[i];

      for (j = 0; j < nPB; ++j)
        xi += Ai[j] * y1_[j];

      x1_[i] = xi;
    }

    /* Second matrix-vector product:
         x2 = x2 + A^T * y_2

       Original code (column-wise access, poor locality):
         for (i)
           for (j)
             x2[i] += A[j][i] * y_2[j];

       We invert the loop order so that A is read row-wise (unit
       stride), which is much more cache-friendly on a row-major
       layout:

         for (j)
           for (i)
             x2[i] += A[j][i] * y_2[j];

       For each fixed i, the sequence of updates to x2[i] still
       follows j = 0,1,2,...,nPB-1, so the sequential result is
       bitwise identical to the original implementation. */
    for (j = 0; j < nPB; ++j)
    {
      const DATA_TYPE y2j = y2_[j];
      const DATA_TYPE * restrict Aj = A_[j];

      for (i = 0; i < nPB; ++i)
        x2_[i] += Aj[i] * y2j;
    }
  }

#pragma endscop

#ifdef _OPENMP
  /* free(NULL) is well-defined, so no explicit null check is needed. */
  free(x2_tmp);
#endif
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