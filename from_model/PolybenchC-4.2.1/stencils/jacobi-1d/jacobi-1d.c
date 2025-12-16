/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"

/* Tunable blocking factor for the spatial dimension.
   Can be overridden at compile time with -DJACOBI_1D_IB=<value>. */
#ifndef JACOBI_1D_IB
# define JACOBI_1D_IB 8192
#endif


/* Array initialization. */
static
void init_array (int n,
                 DATA_TYPE POLYBENCH_1D(A,N,n),
                 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
    {
      A[i] = ((DATA_TYPE) i+ 2) / n;
      B[i] = ((DATA_TYPE) i+ 3) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d[[gnu::flatten, gnu::noinline]](int tsteps,
                                                     int n,
                                                     DATA_TYPE POLYBENCH_1D(A,N,n),
                                                     DATA_TYPE POLYBENCH_1D(B,N,n))
{
  /* Local restrict-qualified aliases help the compiler with
     alias analysis and vectorization. PolyBench allocates A and B
     as distinct arrays, so this is valid. */
  DATA_TYPE * __restrict__ A1 = A;
  DATA_TYPE * __restrict__ B1 = B;

  /* Loop-invariant stencil coefficient (same literal as original). */
  const DATA_TYPE c       = (DATA_TYPE)0.33333;
  const int       n_start = 1;
  const int       n_end   = _PB_N - 1;   /* exclusive upper bound, matches "i < _PB_N - 1" */
  const int       tile_i  = JACOBI_1D_IB;

#pragma scop
  /* One parallel region around the entire time loop amortizes
     fork/join overhead. When compiled without -fopenmp, all
     OpenMP pragmas are ignored and the code executes sequentially
     with the same semantics as the original version. */
#pragma omp parallel default(none) shared(A1,B1,n_start,n_end,tile_i,c)
  {
    int t, ii, i;

    /* Time dimension remains strictly sequential in each thread
       to respect the Jacobi recurrence. Parallelism is exploited
       only in the spatial dimension. */
    for (t = 0; t < _PB_TSTEPS; ++t)
      {
        /* --- Sweep 1: update B from A (A^t -> B^t) --- */
        /* Block the interior spatial domain into tiles of size tile_i
           and distribute tiles across threads. This improves cache
           behavior for very large N and provides a natural grain size
           for parallel work distribution. */
#pragma omp for schedule(static)
        for (ii = n_start; ii < n_end; ii += tile_i)
          {
            const int i_end = (ii + tile_i < n_end) ? (ii + tile_i) : n_end;

            /* Within each tile, iterations over i are contiguous and
               independent; this is ideal for SIMD vectorization. */
#pragma omp simd
            for (i = ii; i < i_end; ++i)
              B1[i] = c * (A1[i-1] + A1[i] + A1[i+1]);
          }

        /* Implicit barrier here ensures all updates to B are visible
           before any thread starts the A-update sweep. */

        /* --- Sweep 2: update A from B (B^t -> A^{t+1}) --- */
#pragma omp for schedule(static)
        for (ii = n_start; ii < n_end; ii += tile_i)
          {
            const int i_end = (ii + tile_i < n_end) ? (ii + tile_i) : n_end;

#pragma omp simd
            for (i = ii; i < i_end; ++i)
              A1[i] = c * (B1[i-1] + B1[i] + B1[i+1]);
          }

        /* Second implicit barrier here: guarantees that the new A
           (time t+1) is fully computed before the next time step. */
      }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}