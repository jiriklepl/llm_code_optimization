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


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied (while preserving the original semantics):

   - Preserve the original loop *order* (k -> i -> j), so all
     data-dependence relations remain identical to the reference code.
   - Split the inner j-loop into three regions: j < k, j == k, and j > k.
     This makes explicit the fact that a[i][k] can change only at j == k,
     and allows us to:
       * Use the "old" a[i][k] value for j < k.
       * Use the (possibly) updated "new" a[i][k] value for j > k.
     This matches exactly the original sequential semantics, including
     cases where a[k][k] may be negative.
   - By doing this split, we:
       * Remove the cross-iteration dependence on a[i][k] inside the
         vectorizable parts (j < k and j > k).
       * Enable efficient reuse of a[i][k] via scalars rather than
         repeatedly loading from memory.
       * Make the j-loops much easier for the compiler to vectorize,
         and we further hint that with `#pragma GCC ivdep`.
   - We also avoid unconditional stores: we only write back to
     path[i][j] when an actual improvement is found, reducing memory
     traffic without changing results.
*/
static
void kernel_floyd_warshall[[gnu::flatten, gnu::noinline]](int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  /* Use the PolyBench-provided runtime loop bound to remain consistent
     with the original code and the benchmarking harness. */
  const int nn = _PB_N;

  /* Local alias to make the code clearer; same object as the parameter. */
  DATA_TYPE (*a)[n] = path;

  int i, j, k;

#pragma scop
  for (k = 0; k < nn; k++)
    {
      /* For each intermediate vertex k, update all pairs (i, j). */
      for (i = 0; i < nn; i++)
      {
        DATA_TYPE *row_i = a[i];
        DATA_TYPE *row_k = a[k];

        /* -------- Part 1: j in [0, k) uses the "old" a[i][k] --------
           In the original code, for all j < k, a[i][k] has not yet been
           updated in this (k, i, *) sweep, so every iteration sees the
           same original value. We load it once into a scalar and reuse. */
        DATA_TYPE a_ik_old = row_i[k];

        /* Guard the first loop so we don't execute it when k == 0. */
        if (k > 0)
        {
#pragma GCC ivdep
          for (j = 0; j < k; j++)
          {
            DATA_TYPE cur = row_i[j];
            DATA_TYPE via = a_ik_old + row_k[j];
            if (via < cur)
              row_i[j] = via;
          }
        }

        /* -------- Part 2: j == k (single iteration) --------
           This is the only iteration that may change a[i][k] itself:
             a[i][k] = min(a[i][k], a[i][k] + a[k][k]);
           We express it explicitly so that Part 1 uses the "old"
           a[i][k], and Part 3 below uses the (possibly) updated one. */
        {
          DATA_TYPE cur = row_i[k];
          DATA_TYPE via = cur + row_k[k];
          if (via < cur)
            row_i[k] = via;
        }

        /* Capture the (possibly) updated value; this matches the value
           that would be seen by all j > k iterations in the original
           sequential loop. */
        DATA_TYPE a_ik_new = row_i[k];

        /* -------- Part 3: j in (k, nn) uses the "new" a[i][k] --------
           In the original loop, all j > k iterations see the value of
           a[i][k] after the j == k update. We again load it once and
           reuse it as a scalar, enabling efficient vectorization. */
        if (k + 1 < nn)
        {
#pragma GCC ivdep
          for (j = k + 1; j < nn; j++)
          {
            DATA_TYPE cur = row_i[j];
            DATA_TYPE via = a_ik_new + row_k[j];
            if (via < cur)
              row_i[j] = via;
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