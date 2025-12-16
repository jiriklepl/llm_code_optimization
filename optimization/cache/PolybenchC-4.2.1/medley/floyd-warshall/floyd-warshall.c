/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"

/* --------------------------------------------------------------------
 * Tunable blocking parameters for the main kernel.
 * They can be overridden at compile time, e.g.:
 *   gcc -DBLOCK_SIZE_I=64 -DBLOCK_SIZE_J=64 ...
 * ------------------------------------------------------------------*/
#ifndef BLOCK_SIZE_I
# define BLOCK_SIZE_I 32
#endif

#ifndef BLOCK_SIZE_J
# define BLOCK_SIZE_J 32
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  /* Use a local restrict-qualified pointer to help the compiler
   * with alias analysis and to make row access explicit. */
  DATA_TYPE (*restrict p)[n] = path;

  for (i = 0; i < n; i++) {
    DATA_TYPE* row = p[i];

    for (j = 0; j < n; j++) {
      DATA_TYPE value = (DATA_TYPE)((i * j) % 7 + 1);
      if (( (i + j) % 13 == 0) ||
          ( (i + j) % 7  == 0) ||
          ( (i + j) % 11 == 0))
        value = (DATA_TYPE)999;
      row[j] = value;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  /* Local pointer for clearer row access. */
  DATA_TYPE (*p)[n] = path;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++) {
    DATA_TYPE* row = p[i];

    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, row[j]);
    }
  }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use a restrict-qualified pointer to the 2D array to improve alias
     analysis.
   - Hoist row pointers and path[i][k] outside the innermost j-loop.
   - Tile (block) the i and j loops for improved cache locality.
   - Preserve the original Floyd–Warshall semantics: k remains the
     outermost loop, while (i,j) are processed in any order for
     each fixed k.
*/
static
void kernel_floyd_warshall[[gnu::flatten, gnu::noinline]](int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

  /* Local restrict-qualified handle to the 2D array.
   * After this point, only 'p' is used to access the matrix,
   * so the restrict contract is respected. */
  DATA_TYPE (*restrict p)[n] = path;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      /* Cache the pointer to row k once per k-iteration. */
      DATA_TYPE *p_k = p[k];

      /* Block the i and j loops to improve cache reuse of rows. */
      for (int ii = 0; ii < _PB_N; ii += BLOCK_SIZE_I)
        {
          int i_end = ii + BLOCK_SIZE_I;
          if (i_end > _PB_N)
            i_end = _PB_N;

          for (int jj = 0; jj < _PB_N; jj += BLOCK_SIZE_J)
            {
              int j_end = jj + BLOCK_SIZE_J;
              if (j_end > _PB_N)
                j_end = _PB_N;

              for (i = ii; i < i_end; ++i)
                {
                  DATA_TYPE *p_i = p[i];
                  /* path[i][k] is invariant across the inner j-loop. */
                  const DATA_TYPE ik = p_i[k];

                  /* The innermost loop walks contiguous memory in row i. */
#pragma GCC ivdep
                  for (j = jj; j < j_end; ++j)
                    {
                      const DATA_TYPE via_k = ik + p_k[j];
                      const DATA_TYPE current = p_i[j];
                      p_i[j] = (current < via_k) ? current : via_k;
                    }
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

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}