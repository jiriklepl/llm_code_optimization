/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"

/* Tunable blocking factors.
 *
 * - BICG_BLOCK_I: tile size in the row dimension (i / N).
 * - BICG_BLOCK_J: tile size in the column dimension (j / M).
 *
 * These are chosen so that a tile of A plus the corresponding slices of
 * p and s fit comfortably in cache. They can be adjusted for a given
 * machine if desired.
 */
#ifndef BICG_BLOCK_I
# define BICG_BLOCK_I 64
#endif

#ifndef BICG_BLOCK_J
# define BICG_BLOCK_J 256
#endif


/* Array initialization. */
static
void init_array (int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                 DATA_TYPE POLYBENCH_1D(r,N,n),
                 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
                 DATA_TYPE POLYBENCH_1D(s,M,m),
                 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg[[gnu::flatten, gnu::noinline]](int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                 DATA_TYPE POLYBENCH_1D(s,M,m),
                 DATA_TYPE POLYBENCH_1D(q,N,n),
                 DATA_TYPE POLYBENCH_1D(p,M,m),
                 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;
  const int tile_i = BICG_BLOCK_I;
  const int tile_j = BICG_BLOCK_J;

  /* Notes on the optimized kernel:
   *
   * Mathematically:
   *   s[j] = sum_i r[i] * A[i][j]   (s = A^T * r)
   *   q[i] = sum_j A[i][j] * p[j]   (q = A * p)
   *
   * The original code already fuses these into a single traversal of A:
   *   for i:
   *     q[i] = 0;
   *     for j:
   *       s[j] += r[i] * A[i][j];
   *       q[i] += A[i][j] * p[j];
   *
   * Optimizations applied here:
   *   - Explicit zero-initialization of s, while q[i] is kept as a
   *     register-resident scalar initialized inside the i-loop.
   *   - 2D tiling of (i, j) into blocks of size tile_i x tile_j to
   *     improve cache locality for A, p, and s.
   *   - Inner loop over j remains the unit-stride dimension to aid
   *     auto-vectorization.
   *   - Manual unrolling by 4 in the j-loop to increase ILP and give
   *     the compiler a simple, straight-line inner body.
   *   - Optional OpenMP parallelization over tiles in the i-dimension,
   *     with an array reduction on s. The sequential execution order
   *     of operations is preserved when OpenMP is not enabled.
   */

#pragma scop
  /* Initialize s exactly as in the original kernel (s[0..M-1] = 0). */
  for (i = 0; i < _PB_M; i++)
    s[i] = SCALAR_VAL(0.0);

  /* Tiled, fused BiCG kernel.
   *
   * Loop structure (sequential semantics):
   *   for ii in 0..N step tile_i
   *     for i in ii..min(ii+tile_i, N)
   *       ri = r[i];
   *       qi = 0;
   *       for jj in 0..M step tile_j
   *         for j in jj..min(jj+tile_j, M)
   *           a = A[i][j];
   *           s[j] += ri * a;
   *           qi   += a * p[j];
   *       q[i] = qi;
   *
   * For each fixed i, j is traversed in ascending order 0..M-1.
   * For each fixed j, i is traversed in ascending order 0..N-1.
   * Thus, the accumulation order into each s[j] and q[i] matches
   * the original scalar implementation.
   */

#ifdef _OPENMP
  /* Parallelize across tiles in the row dimension.
   *
   * - Each iteration of the outer ii-loop works on a distinct block of rows.
   * - q[i] is private to each i and thus each thread (no overlap in i).
   * - s[j] is a reduction over i; OpenMP's array reduction creates
   *   thread-local copies of s[0.._PB_M-1], which are combined at the end.
   *   This adds O(#threads * M) temporary storage, which is modest
   *   compared to the dominant A[N][M] footprint for typical PolyBench sizes.
   */
#pragma omp parallel for schedule(static) private(i, j) reduction(+:s[0:_PB_M])
#endif
  for (int ii = 0; ii < _PB_N; ii += tile_i)
  {
    int i_end = ii + tile_i;
    if (i_end > _PB_N)
      i_end = _PB_N;

    for (i = ii; i < i_end; ++i)
    {
      DATA_TYPE ri = r[i];
      DATA_TYPE qi = SCALAR_VAL(0.0);
      DATA_TYPE *A_row = A[i];

      for (int jj = 0; jj < _PB_M; jj += tile_j)
      {
        int j_end = jj + tile_j;
        if (j_end > _PB_M)
          j_end = _PB_M;

        /* Manually unrolled inner loop over j.
         * Accesses:
         *   - A_row[j]  : unit stride (row-major A)
         *   - p[j], s[j]: unit stride
         * This pattern is very amenable to auto-vectorization.
         */
        j = jj;
        for (; j + 3 < j_end; j += 4)
        {
          DATA_TYPE a0 = A_row[j+0];
          DATA_TYPE a1 = A_row[j+1];
          DATA_TYPE a2 = A_row[j+2];
          DATA_TYPE a3 = A_row[j+3];

          qi      += a0 * p[j+0];
          s[j+0]  += ri * a0;

          qi      += a1 * p[j+1];
          s[j+1]  += ri * a1;

          qi      += a2 * p[j+2];
          s[j+2]  += ri * a2;

          qi      += a3 * p[j+3];
          s[j+3]  += ri * a3;
        }

        /* Remainder loop for the last <4 columns in the tile. */
        for (; j < j_end; ++j)
        {
          DATA_TYPE a_ij = A_row[j];
          qi     += a_ij * p[j];
          s[j]   += ri * a_ij;
        }
      } /* end jj loop */

      q[i] = qi;
    } /* end i loop */
  } /* end ii loop */
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(r),
              POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg (m, n,
               POLYBENCH_ARRAY(A),
               POLYBENCH_ARRAY(s),
               POLYBENCH_ARRAY(q),
               POLYBENCH_ARRAY(p),
               POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}