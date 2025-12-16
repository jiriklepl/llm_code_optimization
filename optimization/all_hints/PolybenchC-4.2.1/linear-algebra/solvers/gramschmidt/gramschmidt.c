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
#include <stdlib.h>  /* for free, size_t */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization.
 *
 * Minor optimizations:
 *  - Use restrict-qualified local pointers to help the compiler
 *    with alias analysis.
 *  - Precompute the scalar factor 100.0 / m to avoid a division
 *    in the inner loop.
 */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict R_)[n] = R;
  DATA_TYPE (*restrict Q_)[n] = Q;

  const DATA_TYPE scale  = SCALAR_VAL(100.0) / (DATA_TYPE)m;
  const DATA_TYPE offset = SCALAR_VAL(10.0);

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      /* Original formula:
         A[i][j] = (((DATA_TYPE)((i*j) % m) / m) * 100) + 10;
         Rewritten as: ((i*j)%m) * (100/m) + 10
         to reduce the cost of division. */
      A_[i][j] = ((DATA_TYPE)((i * j) % m)) * scale + offset;
      Q_[i][j] = SCALAR_VAL(0.0);
    }

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R_[i][j] = SCALAR_VAL(0.0);
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
   including the call and return.

   QR Decomposition with Modified Gram Schmidt:
   http://www.inf.ethz.ch/personal/gander/

   Optimizations applied:

   - Introduce a transposed working copy of A:
       AT[j][i] == A[i][j]
     All operations on columns of A become operations on rows of AT,
     which are contiguous in memory. This greatly improves spatial
     locality and vectorization opportunities for the dominant
     column-wise operations of the algorithm.

     Extra memory: AT has size M*N, which is at most 50% of the
     original footprint (2*M*N + N*N).

   - Use restrict-qualified local pointers (A_, R_, Q_) for the
     PolyBench 2D arrays to help the compiler assume no aliasing.

   - Precompute 1/R[k][k] and use multiplication instead of division
     inside the inner loops.

   - Add OpenMP parallel for pragmas on the inner loops over i (and
     on the transposition loops) to exploit the available cores when
     compiled with OpenMP. When compiled without -fopenmp these
     pragmas are ignored by the compiler and the code remains
     sequential and correct.

   The mathematical algorithm is unchanged; only data layout and loop
   organization are modified. At the end of the kernel, AT is
   transposed back into A so that the externally visible A matches
   the original implementation.
*/
static
void kernel_gramschmidt [[gnu::flatten, gnu::noinline]] (int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

  /* Local restrict-qualified views for better alias analysis. */
  DATA_TYPE (*restrict A_)[n] = A;
  DATA_TYPE (*restrict R_)[n] = R;
  DATA_TYPE (*restrict Q_)[n] = Q;

  /* Transposed working copy of A: AT[j][i] == A[i][j].
     Layout: N rows (columns of A) by M columns (rows of A). */
  DATA_TYPE (*restrict AT)[m] =
    (DATA_TYPE (*)[m]) polybench_alloc_data((size_t)m * (size_t)n,
                                            sizeof(DATA_TYPE));

  /* If allocation fails, the behavior is undefined as in the original
     benchmarks; we assume available memory is sufficient. */

#pragma scop
  /* Initial transpose: AT[j][i] = A[i][j] */
  for (i = 0; i < _PB_M; i++)
    for (j = 0; j < _PB_N; j++)
      AT[j][i] = A_[i][j];

  for (k = 0; k < _PB_N; k++)
    {
      nrm = SCALAR_VAL(0.0);

      /* Compute the squared 2-norm of the k-th column of A.
         Using AT, we read from the contiguous k-th row AT[k][0..M-1]. */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nrm) schedule(static)
#endif
      for (i = 0; i < _PB_M; i++)
      {
        DATA_TYPE val = AT[k][i];
        nrm += val * val;
      }

      R_[k][k] = SQRT_FUN(nrm);

      /* Precompute reciprocal to replace division with multiplication. */
      DATA_TYPE rinv = SCALAR_VAL(1.0) / R_[k][k];

      /* Normalize k-th column of A to produce k-th column of Q. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (i = 0; i < _PB_M; i++)
        Q_[i][k] = AT[k][i] * rinv;

      /* Orthogonalize remaining columns j = k+1 .. N-1. */
      for (j = k + 1; j < _PB_N; j++)
      {
        DATA_TYPE rkj = SCALAR_VAL(0.0);

        /* rkj = dot( Q(:,k), A(:,j) )
           Implemented as dot of contiguous rows:
             Q(:,k) is strided, AT[j][:] is contiguous. */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:rkj) schedule(static)
#endif
        for (i = 0; i < _PB_M; i++)
          rkj += Q_[i][k] * AT[j][i];

        R_[k][j] = rkj;

        /* A(:,j) -= Q(:,k) * rkj
           We update the contiguous row AT[j][:]. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < _PB_M; i++)
          AT[j][i] -= Q_[i][k] * rkj;
      }
    }

  /* Transpose the working copy back so that A matches the original
     kernel's live-out values. */
  for (i = 0; i < _PB_M; i++)
    for (j = 0; j < _PB_N; j++)
      A_[i][j] = AT[j][i];
#pragma endscop

  free(AT);
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