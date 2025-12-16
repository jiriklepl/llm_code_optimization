/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  /* Use local restricted pointers to help the compiler with alias analysis. */
  DATA_TYPE (* restrict A_)[N] = (DATA_TYPE (*)[N]) A;
  DATA_TYPE * restrict b_      = (DATA_TYPE *) b;
  DATA_TYPE * restrict x_      = (DATA_TYPE *) x;
  DATA_TYPE * restrict y_      = (DATA_TYPE *) y;

  /* Initialize vectors b, x, y. */
  for (i = 0; i < n; i++)
    {
      x_[i] = (DATA_TYPE)0;
      y_[i] = (DATA_TYPE)0;
      b_[i] = (DATA_TYPE)((i+1)/fn/2.0 + 4);
    }

  /* Initialize A as a unit-lower-triangular-like matrix. */
  for (i = 0; i < n; i++)
    {
      DATA_TYPE *Ai = A_[i];
      for (j = 0; j <= i; j++)
	Ai[j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++)
	Ai[j] = (DATA_TYPE)0;
      Ai[i] = (DATA_TYPE)1;
    }

  /* Make the matrix positive semi-definite.
   *
   * Original code computed:
   *   B[r][s] = sum_t A[r][t] * A[s][t]
   * with three nested loops and an explicit initialization of B to zero.
   *
   * Here we compute the same Gram matrix more efficiently by:
   *   - Exploiting symmetry: B[r][s] == B[s][r], so we only compute one
   *     dot product per (r,s) pair with r <= s, and mirror the result.
   *   - Using contiguous row access (r, s, t loop order) for better
   *     cache locality.
   *
   * For each (r,s), the accumulation is still performed in increasing
   * order of t, so the sequence of floating-point operations for that
   * entry is identical to the original, ensuring bitwise-identical
   * results for all B[r][s] (and thus for A after the copy-back).
   */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  DATA_TYPE (* restrict B_)[N] = (DATA_TYPE (*)[N]) POLYBENCH_ARRAY(B);

  for (r = 0; r < n; ++r)
    {
      const DATA_TYPE * restrict Ar = A_[r];
      for (s = r; s < n; ++s)
        {
          const DATA_TYPE * restrict As = A_[s];
          DATA_TYPE sum = (DATA_TYPE)0;

          /* Dot product of rows r and s. */
          for (t = 0; t < n; ++t)
            sum += Ar[t] * As[t];

          B_[r][s] = sum;
          if (r != s)
            B_[s][r] = sum; /* exploit symmetry */
        }
    }

  /* Copy back to A. */
  for (r = 0; r < n; ++r)
    {
      DATA_TYPE * restrict Ar = A_[r];
      for (s = 0; s < n; ++s)
	Ar[s] = B_[r][s];
    }

  POLYBENCH_FREE_ARRAY(B);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp [[gnu::flatten, gnu::noinline]] (int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;
  DATA_TYPE w;

  /* Local restricted views of the input arrays to improve alias analysis
   * and enable better vectorization and instruction scheduling. */
  DATA_TYPE (* restrict A_)[N] = (DATA_TYPE (*)[N]) A;
  DATA_TYPE * restrict b_      = (DATA_TYPE *) b;
  DATA_TYPE * restrict x_      = (DATA_TYPE *) x;
  DATA_TYPE * restrict y_      = (DATA_TYPE *) y;

#pragma scop
  /* LU factorization without pivoting. */
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE *Ai = A_[i];

    /* Compute L(i, 0..i-1) */
    for (j = 0; j < i; j++) {
       DATA_TYPE *Aj = A_[j]; /* row j of A */
       w = Ai[j];

       /* Inner product: subtract sum_{k=0..j-1} A[i][k] * A[k][j] */
       for (k = 0; k < j; k++) {
          w -= Ai[k] * Aj[k];
       }
       Ai[j] = w / Aj[j];
    }

    /* Compute U(i, i..N-1) */
    for (j = i; j < _PB_N; j++) {
       DATA_TYPE *Aj = A_[j]; /* row j of A */
       w = Ai[j];
       for (k = 0; k < i; k++) {
          w -= Ai[k] * Aj[k];
       }
       Ai[j] = w;
    }
  }

  /* Forward substitution: solve L * y = b. */
  for (i = 0; i < _PB_N; i++) {
     DATA_TYPE *Ai = A_[i];
     w = b_[i];
     for (j = 0; j < i; j++)
        w -= Ai[j] * y_[j];
     y_[i] = w;
  }

  /* Backward substitution: solve U * x = y. */
  for (i = _PB_N-1; i >=0; i--) {
     DATA_TYPE *Ai = A_[i];
     w = y_[i];
     for (j = i+1; j < _PB_N; j++)
        w -= Ai[j] * x_[j];
     x_[i] = w / Ai[i];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}