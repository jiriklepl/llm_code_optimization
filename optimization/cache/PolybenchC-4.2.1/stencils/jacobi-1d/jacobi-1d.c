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

/* --------------------------------------------------------------------------
 * Tunable kernel parameters
 * --------------------------------------------------------------------------
 * JACOBI_1D_UNROLL_FACTOR:
 *   Hint for the compiler about how aggressively to unroll the main
 *   spatial loops. Can be overridden at compile time, e.g.:
 *
 *     gcc -O3 -DNDEBUG -DJACOBI_1D_UNROLL_FACTOR=4 ...
 *
 *   This does not change semantics; it only influences code generation.
 */
#ifndef JACOBI_1D_UNROLL_FACTOR
# define JACOBI_1D_UNROLL_FACTOR 8
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

/* --------------------------------------------------------------------------
 * Optimized inner Jacobi step
 * --------------------------------------------------------------------------
 * Computes one Jacobi time step:
 *
 *   dst[i] = 0.33333 * (src[i-1] + src[i] + src[i+1]),  for 1 <= i < n-1
 *
 * while keeping dst[0] and dst[n-1] unchanged (they are never written),
 * exactly as in the original kernel.
 *
 * Optimization techniques:
 *  - "Sliding window" over src: each element of src is loaded once and
 *    reused from registers for the next two stencil points. This reduces
 *    memory traffic and improves cache utilization compared to directly
 *    reading src[i-1], src[i], src[i+1] in each iteration.
 *  - restrict-qualified pointers: inform the compiler that src and dst
 *    do not alias, enabling better vectorization and reordering.
 *  - Unrolling and vectorization hints via GCC pragmas.
 */
static inline void
jacobi_step_1d(int n,
               const DATA_TYPE * __restrict src,
               DATA_TYPE       * __restrict dst,
               const DATA_TYPE  coeff)
{
  /* No interior points if n <= 2, nothing to do.
     Matches the original loops which run for i = 1 .. n-2. */
  if (n <= 2)
    return;

  int i;
  DATA_TYPE prev = src[0];
  DATA_TYPE curr = src[1];

  /* Hint to GCC: loop has no cross-iteration dependencies on dst,
     so it is safe to vectorize. */
#pragma GCC ivdep
#pragma GCC unroll JACOBI_1D_UNROLL_FACTOR
  for (i = 1; i < n - 1; i++)
    {
      DATA_TYPE next = src[i+1];
      /* Stencil computation with a 3-point sliding window. */
      dst[i] = coeff * (prev + curr + next);

      /* Advance the window for the next iteration. */
      prev = curr;
      curr = next;
    }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __attribute__((noinline))
kernel_jacobi_1d(int tsteps,
                 int n,
                 DATA_TYPE POLYBENCH_1D(A,N,n),
                 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  /* Local aliases marked restrict to help the optimizer.
     A and B are allocated as separate arrays, so this is valid. */
  DATA_TYPE * __restrict a = A;
  DATA_TYPE * __restrict b = B;

  /* PolyBench loop bounds (may differ from raw n / tsteps in some modes). */
  const int T = _PB_TSTEPS;
  const int Nloc = _PB_N;

  /* Stencil coefficient, kept in a register across the whole kernel. */
  const DATA_TYPE coeff = (DATA_TYPE)0.33333;

  int t;

#pragma scop
  /* If there are no interior points, the original code's loops are empty.
     We preserve that behavior explicitly. */
  if (Nloc > 2)
    {
      for (t = 0; t < T; t++)
        {
          /* Jacobi update: A -> B, then B -> A. */
          jacobi_step_1d(Nloc, a, b, coeff);
          jacobi_step_1d(Nloc, b, a, coeff);
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