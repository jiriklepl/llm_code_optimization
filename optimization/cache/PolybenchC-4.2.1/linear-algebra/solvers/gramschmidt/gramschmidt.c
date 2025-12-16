/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"

/* Tunable blocking factor along the column dimension (j).
   Can be changed at compile time, e.g.,
   -DGRAMSCHMIDT_J_BLOCK=128 */
#ifndef GRAMSCHMIDT_J_BLOCK
# define GRAMSCHMIDT_J_BLOCK 64
#endif


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt[[gnu::flatten, gnu::noinline]](int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  /* Create restrict-qualified aliases with an explicit second dimension.
     PolyBench guarantees that A, R, and Q do not alias, which enables
     stronger optimizations (vectorization, hoisting, etc.). */
  DATA_TYPE (* restrict A_)[n] = A;
  DATA_TYPE (* restrict R_)[n] = R;
  DATA_TYPE (* restrict Q_)[n] = Q;

  /* Tunable j-block size for improved cache locality when traversing
     the column space. */
  const int j_block_size = GRAMSCHMIDT_J_BLOCK;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      /* 1. Compute the 2-norm of the k-th column of A. */
      DATA_TYPE nrm = SCALAR_VAL(0.0);

      for (i = 0; i < _PB_M; i++)
	{
	  DATA_TYPE aik = A_[i][k];
	  nrm += aik * aik;
	}

      DATA_TYPE rkk = SQRT_FUN(nrm);
      R_[k][k] = rkk;

      /* Use a reciprocal to replace divisions with multiplications. */
      DATA_TYPE inv_rkk = SCALAR_VAL(1.0) / rkk;

      /* 2. Form the k-th column of Q: Q[:,k] = A[:,k] / R[k][k]. */
      for (i = 0; i < _PB_M; i++)
	{
	  Q_[i][k] = A_[i][k] * inv_rkk;
	}

      /* 3. Initialize the upper-triangular part of the k-th row of R. */
      for (j = k + 1; j < _PB_N; j++)
	{
	  R_[k][j] = SCALAR_VAL(0.0);
	}

      /* 4. Compute R[k][j] = Q[:,k]^T * A[:,j] for all j > k.
	 Original code:
	     for j>k: R[k][j] = sum_i Q[i][k] * A[i][j];

	 Here we reorder the loops to have i outer and j inner, which:
	   - traverses A[i][j] and R[k][j] with unit stride in j,
	   - loads Q[i][k] once per i and reuses it across all j.

	 The accumulation order for each R[k][j] over i is preserved,
	 so floating-point results remain bitwise identical. */
      for (i = 0; i < _PB_M; i++)
	{
	  DATA_TYPE qik = Q_[i][k];

	  int j_end = _PB_N;
	  int jj;

	  for (jj = k + 1; jj < j_end; jj += j_block_size)
	    {
	      int upper = jj + j_block_size;
	      if (upper > j_end)
		upper = j_end;

	      /* Inner j-loop operates on contiguous memory and has
		 no loop-carried dependencies → good for SIMD. */
#pragma GCC ivdep
	      for (j = jj; j < upper; j++)
		{
		  R_[k][j] += qik * A_[i][j];
		}
	    }
	}

      /* 5. Update the remaining columns of A:
	     A[:,j] = A[:,j] - Q[:,k] * R[k][j],  for j > k.

	 Loop ordering (i outer, j inner) again gives unit-stride access
	 in j for both A and R, and reuses Q[i][k] across many columns.
	 Each A[i][j] is updated exactly once, so the semantics match
	 the original kernel. */
      for (i = 0; i < _PB_M; i++)
	{
	  DATA_TYPE qik = Q_[i][k];

	  int j_end = _PB_N;
	  int jj;

	  for (jj = k + 1; jj < j_end; jj += j_block_size)
	    {
	      int upper = jj + j_block_size;
	      if (upper > j_end)
		upper = j_end;

#pragma GCC ivdep
	      for (j = jj; j < upper; j++)
		{
		  A_[i][j] -= qik * R_[k][j];
		}
	    }
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
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}