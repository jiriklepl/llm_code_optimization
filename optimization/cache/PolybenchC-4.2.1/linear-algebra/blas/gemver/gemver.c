/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

/* Tunable blocking factor for the w-update in kernel_gemver.
 * Override at compile time with:  -DGEMVER_W_J_BLOCK=<positive integer>
 * The default (64) is a reasonable choice for typical x64 L1/L2 sizes.
 */
#ifndef GEMVER_W_J_BLOCK
# define GEMVER_W_J_BLOCK 64
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      /* Compute (i+1)/fn once and reuse for the different scalings.
         This preserves the original arithmetic order for each element. */
      DATA_TYPE t = ((DATA_TYPE)(i + 1)) / fn;

      u1[i] = i;
      u2[i] = t / 2.0;
      v1[i] = t / 4.0;
      v2[i] = t / 6.0;
      y[i]  = t / 8.0;
      z[i]  = t / 9.0;

      x[i] = 0.0;
      w[i] = 0.0;

      /* Access A row-wise for better spatial locality. */
      DATA_TYPE *Ai = A[i];
      for (j = 0; j < n; j++)
        Ai[j] = (DATA_TYPE) (i * j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver[[gnu::flatten, gnu::noinline]](int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;
  /* Cache the PolyBench-size macro in a local variable. */
  int nPB = _PB_N;

#pragma scop

  /* 1. Rank-1 updates: A += u1 * v1^T + u2 * v2^T
     Keep i outer / j inner so that the innermost loop walks contiguously
     in memory. Hoist u1[i] and u2[i] out of the j-loop to reduce loads. */
  for (i = 0; i < nPB; i++)
    {
      DATA_TYPE ui1 = u1[i];
      DATA_TYPE ui2 = u2[i];
      DATA_TYPE *Ai = A[i]; /* row pointer to help auto-vectorization */

      for (j = 0; j < nPB; j++)
        Ai[j] = Ai[j] + ui1 * v1[j] + ui2 * v2[j];
    }

  /* 2. x = x + beta * A^T * y
     Original code:
       for (i)
         for (j)
           x[i] += beta * A[j][i] * y[j];

     That traverses A column-wise in the inner loop, which is cache-unfriendly.
     We interchange the loops so that the innermost loop walks A row-wise.
     For each fixed i, the updates to x[i] still see j in ascending order
     (0..nPB-1), so the accumulation order per element is preserved. */
  for (j = 0; j < nPB; j++)
    {
      DATA_TYPE yj = beta * y[j];
      DATA_TYPE *Aj = A[j]; /* row j of A */

      for (i = 0; i < nPB; i++)
        x[i] = x[i] + Aj[i] * yj;
    }

  /* 3. x = x + z (simple vector update). */
  for (i = 0; i < nPB; i++)
    x[i] = x[i] + z[i];

  /* 4. w = w + alpha * A * x
     This loop already accesses A row-wise (i outer / j inner). We retain
     that structure and add a tunable blocking factor on j to better utilize
     caches for large problem sizes, while preserving the original order of
     accumulation for each w[i]. */
  for (i = 0; i < nPB; i++)
    {
      DATA_TYPE wi = w[i];
      DATA_TYPE *Ai = A[i]; /* row i of A */

      for (int jj = 0; jj < nPB; jj += GEMVER_W_J_BLOCK)
        {
          int j_end = jj + GEMVER_W_J_BLOCK;
          if (j_end > nPB)
            j_end = nPB;

          /* Inner loop walks contiguous slices of A[i][*] and x[*]. */
          for (j = jj; j < j_end; j++)
            wi = wi + alpha * Ai[j] * x[j];
        }

      w[i] = wi;
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
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}