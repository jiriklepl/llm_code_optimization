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
#include <stdlib.h>  /* for malloc/free */

#ifdef _OPENMP
# include <omp.h>    /* for omp_get_max_threads / omp_get_thread_num */
#endif

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization.
 *
 * Minor optimizations:
 *  - Precompute the denominator used for initializing A.
 *  - Use SCALAR_VAL to keep types consistent.
 *  - Optionally allow OpenMP parallelization of the loops (ignored if
 *    compiled without -fopenmp).
 */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  const DATA_TYPE fn    = (DATA_TYPE) n;
  const DATA_TYPE denom = SCALAR_VAL(5.0) * (DATA_TYPE) m;

  /* Initialize x: x[i] = 1 + i / fn */
#pragma omp parallel for if (n > 1024) schedule(static)
  for (i = 0; i < n; i++)
    x[i] = SCALAR_VAL(1.0) + ((DATA_TYPE) i / fn);

  /* Initialize A: A[i][j] = ((i + j) % n) / (5 * m) */
#pragma omp parallel for if ((long long)m * (long long)n > 4096) schedule(static)
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE)((i + j) % n)) / denom;
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
 * Optimizations applied:
 *  - Keep per-row temporary in a scalar (tmp_i) and write tmp[i] only
 *    once per row instead of every inner-iteration.
 *  - Add local restrict-qualified aliases for x, y, tmp to help the
 *    compiler prove non-aliasing and enable better vectorization.
 *  - Preserve row-major traversal of A for good spatial locality.
 *  - Provide an OpenMP parallel implementation that:
 *      * Parallelizes over the row index i.
 *      * Uses a per-thread private y buffer to avoid write conflicts.
 *      * Combines partial y vectors at the end.
 *    The per-thread buffer size is limited so that the extra memory
 *    usage never exceeds 50% of the original data footprint.
 *  - When OpenMP is not enabled, the code falls back to an optimized
 *    sequential implementation; OpenMP pragmas are ignored by the
 *    compiler in that case.
 */
static
void kernel_atax [[gnu::flatten, gnu::noinline]] (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  /* Use the PolyBench loop bounds so that any build-time tuning done
     by PolyBench (e.g., via command-line defines) is preserved. */
  const int Mdim = _PB_M;
  const int Ndim = _PB_N;

  /* Local restrict-qualified aliases help the optimizer reason about
     aliasing between the main arrays.  We never use the original
     parameter names after this point. */
  DATA_TYPE * restrict x_   = x;
  DATA_TYPE * restrict y_   = y;
  DATA_TYPE * restrict tmp_ = tmp;

  (void)m; /* parameters kept for interface compatibility */
  (void)n;

#pragma scop
  /* Initialize y to zero. This is kept as a separate loop so that
     compilers can easily vectorize / parallelize it. */
#pragma omp parallel for if (Ndim > 1024) schedule(static)
  for (j = 0; j < Ndim; j++)
    y_[j] = SCALAR_VAL(0.0);

  /* Decide how many threads to use for the threaded implementation.
     When OpenMP is not available, num_threads remains 1.
     We also cap the number of threads to ensure the extra memory
     used for per-thread y buffers is at most 50% of the original
     data footprint.

     Original data footprint in elements (ignoring small constant
     overheads) is approximately:
        base_elems = M*N + 2*N + M   (A, x, y, tmp).
     Extra elements for private y buffers:
        extra_elems = T * N.

     We enforce: extra_elems <= 0.5 * base_elems
     -> T <= base_elems / (2 * N).
  */
  int num_threads = 1;
#ifdef _OPENMP
  {
    int omp_max = omp_get_max_threads();
    if (omp_max < 1)
      omp_max = 1;

    size_t base_elems =
      (size_t)Mdim * (size_t)Ndim +  /* A */
      (size_t)2 * (size_t)Ndim +     /* x and y */
      (size_t)Mdim;                  /* tmp */

    int mem_limit_threads = 1;
    if (Ndim > 0) {
      mem_limit_threads = (int)(base_elems / (2u * (size_t)Ndim));
      if (mem_limit_threads < 1)
        mem_limit_threads = 1;
    }

    num_threads = (omp_max < mem_limit_threads) ? omp_max : mem_limit_threads;
  }
#endif /* _OPENMP */

  /* Optimized sequential path (also used as a fallback if threading
     is not possible or memory allocation for private buffers fails). */
  if (num_threads <= 1)
  {
    for (i = 0; i < Mdim; i++)
    {
      DATA_TYPE tmp_i = SCALAR_VAL(0.0);
      DATA_TYPE * restrict Arow = A[i];

      /* First pass: tmp_i = dot(A[i][:], x[:]) */
      for (j = 0; j < Ndim; j++)
        tmp_i += Arow[j] * x_[j];

      tmp_[i] = tmp_i;

      /* Second pass: y[:] += A[i][:] * tmp_i */
      for (j = 0; j < Ndim; j++)
        y_[j] += Arow[j] * tmp_i;
    }
  }
  else
  {
    /* Threaded implementation with manually privatized y. */

    /* Allocate a single contiguous buffer that will hold num_threads
       private copies of y.  Layout is:
           y_priv[0][0..Ndim-1],
           y_priv[1][0..Ndim-1],
           ...
       where y_priv[t] is used by thread t.
    */
    const size_t priv_elems = (size_t)num_threads * (size_t)Ndim;
    DATA_TYPE *y_priv = (DATA_TYPE *) malloc(priv_elems * sizeof(DATA_TYPE));

    if (y_priv == NULL)
    {
      /* Allocation failed: fall back to the optimized sequential
         implementation while preserving semantics. */
      for (i = 0; i < Mdim; i++)
      {
        DATA_TYPE tmp_i = SCALAR_VAL(0.0);
        DATA_TYPE * restrict Arow = A[i];

        for (j = 0; j < Ndim; j++)
          tmp_i += Arow[j] * x_[j];

        tmp_[i] = tmp_i;

        for (j = 0; j < Ndim; j++)
          y_[j] += Arow[j] * tmp_i;
      }
    }
    else
    {
      /* Initialize all private y buffers to zero.
         This is O(num_threads * Ndim), which is negligible compared
         to the O(Mdim * Ndim) main computation. */
      for (size_t idx = 0; idx < priv_elems; idx++)
        y_priv[idx] = SCALAR_VAL(0.0);

      /* Parallel region: distribute the outer loop over rows i
         across threads. Each thread uses its own y_private buffer,
         removing any need for atomics or critical sections on y. */
#pragma omp parallel num_threads(num_threads)
      {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        DATA_TYPE * restrict y_local =
          y_priv + (size_t)tid * (size_t)Ndim;

#pragma omp for schedule(static)
        for (i = 0; i < Mdim; i++)
        {
          DATA_TYPE tmp_i = SCALAR_VAL(0.0);
          DATA_TYPE * restrict Arow = A[i];

          /* tmp_i = A[i][:] * x[:] */
          for (j = 0; j < Ndim; j++)
            tmp_i += Arow[j] * x_[j];

          tmp_[i] = tmp_i;

          /* Accumulate into this thread's private y buffer. */
          for (j = 0; j < Ndim; j++)
            y_local[j] += Arow[j] * tmp_i;
        }
      } /* end parallel region */

      /* Combine per-thread partial y vectors into the final result y.
         This reduction is performed sequentially; its cost
         num_threads * Ndim is small relative to the main kernel. */
      for (int t = 0; t < num_threads; t++)
      {
        DATA_TYPE * restrict y_local =
          y_priv + (size_t)t * (size_t)Ndim;
        for (j = 0; j < Ndim; j++)
          y_[j] += y_local[j];
      }

      free(y_priv);
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