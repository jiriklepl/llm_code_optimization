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
   including the call and return. */
static
void kernel_atax[[gnu::flatten, gnu::noinline]](int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  /* --------------------------------------------------------------------
   * Local optimized views of the input arrays.
   *
   *  - We add 'restrict' to tell the compiler that these pointers do not
   *    alias each other.  In PolyBench, A/x/y/tmp are allocated from
   *    distinct memory regions, so this is a valid assumption and enables
   *    better vectorization and scheduling.
   *
   *  - We also communicate the alignment guaranteed by PolyBench
   *    (POLYBENCH_ALIGNMENT bytes) via __builtin_assume_aligned, which
   *    allows the compiler to generate aligned SIMD loads/stores when
   *    profitable.
   *
   *  These transformations do not change the program's semantics; they
   *  only give the optimizer more precise information about the data.
   * ------------------------------------------------------------------ */
  DATA_TYPE (*restrict A_restrict)[n] =
    (DATA_TYPE (*restrict)[n])__builtin_assume_aligned(A, POLYBENCH_ALIGNMENT);
  DATA_TYPE *restrict x_restrict =
    (DATA_TYPE *restrict)__builtin_assume_aligned(x, POLYBENCH_ALIGNMENT);
  DATA_TYPE *restrict y_restrict =
    (DATA_TYPE *restrict)__builtin_assume_aligned(y, POLYBENCH_ALIGNMENT);
  DATA_TYPE *restrict tmp_restrict =
    (DATA_TYPE *restrict)__builtin_assume_aligned(tmp, POLYBENCH_ALIGNMENT);

#pragma scop
  /* y[:] = 0 */
  for (i = 0; i < _PB_N; i++)
    y_restrict[i] = SCALAR_VAL(0.0);

  /* Core ATAX computation:
   *
   *   tmp[i] = A[i,:] * x          (matrix-vector product)
   *   y     += tmp[i] * A[i,:]^T   (rank-1 update of y)
   *
   * We keep the original mathematical order of operations:
   *  - tmp[i] is fully computed before being used to update y.
   *  - For each fixed j, the sequence of additions contributing to y[j]
   *    is still performed in increasing i order.
   *
   * Optimizations inside the loop:
   *  - Keep tmp[i] in a scalar register (tmp_i) and write it to tmp[]
   *    only once per row, instead of reloading it from memory.
   *  - Use a row pointer Ai = A_restrict[i] to avoid repeated address
   *    arithmetic and to keep accesses to A row-contiguous.
   *  - Manually unroll the inner j-loops by a factor of 4 to expose more
   *    instruction-level parallelism and help the compiler generate
   *    efficient SIMD code.
   */
  for (i = 0; i < _PB_M; i++)
    {
      DATA_TYPE *restrict Ai = A_restrict[i];
      DATA_TYPE tmp_i = SCALAR_VAL(0.0);

      /* Compute tmp_i = dot(A[i][:], x[:]) */
      {
        int j_unroll = _PB_N & ~3; /* largest multiple of 4 <= _PB_N */

        for (j = 0; j < j_unroll; j += 4)
          {
            tmp_i += Ai[j    ] * x_restrict[j    ];
            tmp_i += Ai[j + 1] * x_restrict[j + 1];
            tmp_i += Ai[j + 2] * x_restrict[j + 2];
            tmp_i += Ai[j + 3] * x_restrict[j + 3];
          }
        /* Remainder loop for N not divisible by 4 */
        for (; j < _PB_N; j++)
          tmp_i += Ai[j] * x_restrict[j];
      }

      /* Store the final tmp value for this row (live-out array). */
      tmp_restrict[i] = tmp_i;

      /* y[:] += tmp_i * A[i][:] */
      {
        int j_unroll = _PB_N & ~3;

        for (j = 0; j < j_unroll; j += 4)
          {
            const DATA_TYPE t = tmp_i; /* keep in a register */
            y_restrict[j    ] += Ai[j    ] * t;
            y_restrict[j + 1] += Ai[j + 1] * t;
            y_restrict[j + 2] += Ai[j + 2] * t;
            y_restrict[j + 3] += Ai[j + 3] * t;
          }
        /* Remainder loop */
        for (; j < _PB_N; j++)
          y_restrict[j] += Ai[j] * tmp_i;
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