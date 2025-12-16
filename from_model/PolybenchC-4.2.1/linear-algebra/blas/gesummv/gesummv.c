/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
    }
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
void kernel_gesummv[[gnu::flatten, gnu::noinline]](int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
#pragma scop
  /* Tunable blocking / unrolling parameters.
     These are conservative defaults for modern x86-64 cores. */
  const int Ti       = 64;   /* tile size for the outer i-dimension  */
  const int Tj       = 256;  /* tile size for the inner j-dimension  */
  const int UNROLL_J = 4;    /* unroll factor for the inner j-loop   */

  const int n_pb = _PB_N;

  /* Local restrict-qualified views of the inputs/outputs.
     This matches the actual program behavior (distinct PolyBench
     allocations) and helps the compiler with alias analysis and
     vectorization. */
  DATA_TYPE (*restrict A_)[n_pb] = A;
  DATA_TYPE (*restrict B_)[n_pb] = B;
  DATA_TYPE *restrict tmp_       = tmp;
  DATA_TYPE *restrict y_         = y;
  const DATA_TYPE *restrict x_   = x;

  /* Tile over rows (i) to provide a coarse grain of work that can
     be scheduled efficiently across cores if OpenMP is enabled. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int ii = 0; ii < n_pb; ii += Ti)
    {
      const int i_max = (ii + Ti < n_pb) ? (ii + Ti) : n_pb;

      for (int i = ii; i < i_max; ++i)
        {
          /* Row-local accumulators kept in registers.
             These play the role of tmp[i] and the running sum of B*x. */
          DATA_TYPE sumA = SCALAR_VAL(0.0);
          DATA_TYPE sumB = SCALAR_VAL(0.0);

          /* Pointers to the current rows of A and B (unit-stride in j). */
          DATA_TYPE *restrict Ai = A_[i];
          DATA_TYPE *restrict Bi = B_[i];

          /* Tile the j-dimension to keep a bounded working set of A, B,
             and x in the L1/L2 caches for very large N. */
          for (int jj = 0; jj < n_pb; jj += Tj)
            {
              const int j_max_tile = (jj + Tj < n_pb) ? (jj + Tj) : n_pb;

              /* Manually unrolled inner loop over j for ILP and to
                 encourage SIMD vectorization.  The accumulation order
                 over j is identical to the original kernel:
                   j = 0,1,2,...,N-1. */
              int j = jj;
              const int span   = j_max_tile - jj;
              const int j_stop = j_max_tile - (span % UNROLL_J);

              for (; j < j_stop; j += UNROLL_J)
                {
                  DATA_TYPE x0 = x_[j];
                  sumA += Ai[j] * x0;
                  sumB += Bi[j] * x0;

                  DATA_TYPE x1 = x_[j + 1];
                  sumA += Ai[j + 1] * x1;
                  sumB += Bi[j + 1] * x1;

                  DATA_TYPE x2 = x_[j + 2];
                  sumA += Ai[j + 2] * x2;
                  sumB += Bi[j + 2] * x2;

                  DATA_TYPE x3 = x_[j + 3];
                  sumA += Ai[j + 3] * x3;
                  sumB += Bi[j + 3] * x3;
                }

              /* Handle the remaining (< UNROLL_J) elements in this tile. */
              for (; j < j_max_tile; ++j)
                {
                  DATA_TYPE xj = x_[j];
                  sumA += Ai[j] * xj;
                  sumB += Bi[j] * xj;
                }
            }

          /* Finalize outputs for row i.
             tmp[i] is materialized once, and y[i] uses the same
             values as in the original code:
               tmp[i] = sum_j A[i,j] * x[j]
               y[i]   = alpha * tmp[i] + beta * sum_j B[i,j] * x[j]
           */
          tmp_[i] = sumA;
          y_[i]   = alpha * sumA + beta * sumB;
        }
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gesummv (n, alpha, beta,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(B),
		  POLYBENCH_ARRAY(tmp),
		  POLYBENCH_ARRAY(x),
		  POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}