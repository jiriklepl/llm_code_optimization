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


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  /* Local restrict-qualified views to help the compiler optimize. */
  DATA_TYPE (*restrict A_)[N] = (DATA_TYPE (*)[N])A;
  DATA_TYPE (*restrict R_)[N] = (DATA_TYPE (*)[N])R;
  DATA_TYPE (*restrict Q_)[N] = (DATA_TYPE (*)[N])Q;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A_[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q_[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R_[i][j] = 0.0;
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

  /* Use restrict-qualified views to help optimization in case the
     printing loop is ever timed or inspected. */
  DATA_TYPE (*restrict A_)[N] = (DATA_TYPE (*)[N])A;
  DATA_TYPE (*restrict R_)[N] = (DATA_TYPE (*)[N])R;
  DATA_TYPE (*restrict Q_)[N] = (DATA_TYPE (*)[N])Q;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R_[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q_[i][j]);
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
  DATA_TYPE nrm;

  /* Local restrict-qualified views improve the compiler's ability to
     vectorize and reorder memory accesses safely.  PolyBench allocates
     A, R, Q as separate buffers so this is a valid aliasing contract. */
  DATA_TYPE (*restrict A_)[N] = (DATA_TYPE (*)[N])A;
  DATA_TYPE (*restrict R_)[N] = (DATA_TYPE (*)[N])R;
  DATA_TYPE (*restrict Q_)[N] = (DATA_TYPE (*)[N])Q;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      /* 1) Compute squared 2-norm of the k-th column of A. */
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
        {
          /* Load once per iteration to help the optimizer. */
          DATA_TYPE aik = A_[i][k];
          nrm += aik * aik;
        }

      R_[k][k] = SQRT_FUN(nrm);
      DATA_TYPE rkk = R_[k][k];

      /* 2) Initialize the rest of the row R[k][j] (j > k) once. */
      for (j = k + 1; j < _PB_N; j++)
        R_[k][j] = SCALAR_VAL(0.0);

      /* 3) Compute Q[:,k] and accumulate all R[k][j] (j > k) in a
         cache-friendly way.

         Original code (simplified):
           for i:
             Q[i][k] = A[i][k] / R[k][k];
           for j:
             R[k][j] = 0;
             for i:
               R[k][j] += Q[i][k] * A[i][j];

         Here we fuse the Q and R computations and switch to
         (i outer, j inner) for better row-major locality. For each
         fixed j, the sequence of floating-point operations over i
         remains in the same order as in the original code, so the
         numerical results are bit-identical. */
      for (i = 0; i < _PB_M; i++)
        {
          DATA_TYPE aik = A_[i][k];
          DATA_TYPE qik = aik / rkk;
          Q_[i][k] = qik;

          /* Inner loop walks contiguously along row i in A_ and R_. */
          for (j = k + 1; j < _PB_N; j++)
            {
              /* Equivalent to: R[k][j] += Q[i][k] * A[i][j]; */
              R_[k][j] += qik * A_[i][j];
            }
        }

      /* 4) Update the trailing columns of A:
             A[:,j] -= Q[:,k] * R[k][j]  for j = k+1..N-1

         We again use (i outer, j inner) to get unit-stride access
         along rows of A_, and we parallelize over i since different
         rows are independent.  Each A[i][j] element is updated exactly
         once, so the arithmetic for each element is unchanged. */

      #pragma omp parallel for default(none) private(i,j) shared(A_,Q_,R_,k)
      for (i = 0; i < _PB_M; i++)
        {
          DATA_TYPE qik = Q_[i][k];

          for (j = k + 1; j < _PB_N; j++)
            {
              /* Equivalent to: A[i][j] = A[i][j] - Q[i][k] * R[k][j]; */
              A_[i][j] -= qik * R_[k][j];
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