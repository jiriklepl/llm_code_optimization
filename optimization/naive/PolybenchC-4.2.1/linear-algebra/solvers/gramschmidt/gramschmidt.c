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
void kernel_gramschmidt [[gnu::flatten, gnu::noinline]] (int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  (void)m; (void)n; /* Parameters are unused; problem sizes are given by macros. */

  int i, j, k;
  DATA_TYPE nrm;

  /*
   * Optimization notes:
   * - Create local 'restrict' aliases for A, R, Q to tell the compiler
   *   that these arrays do not alias.  This enables stronger
   *   vectorization and hoisting of loads.
   * - Use local constants M_ and N_ as loop bounds; they are compile-time
   *   constants derived from PolyBench macros (_PB_M/_PB_N).
   * - Reorder the innermost loops that compute R[k][j] and update A[i][j]
   *   so that the innermost loop iterates over j.  Since the arrays are
   *   stored in row-major order, this makes the innermost loop access
   *   memory contiguously, improving cache and TLB behavior.
   * - In the original code, for each (k,j) pair R[k][j] is computed as a
   *   dot product over i, and A[i][j] is updated in a separate loop over i.
   *   The transformed version preserves the exact per-(k,j) reduction
   *   order over i, but computes all R[k][j] for fixed k in a single pass
   *   over rows.  This also allows each Q[i][k] value to be reused across
   *   all j, significantly reducing redundant loads.
   * - Replace division by R[k][k] inside the i-loop by a single
   *   reciprocal computed once per k and used as a multiplication.
   */

  /* Local restricted aliases (non-aliasing promises are valid for this benchmark). */
  DATA_TYPE (* restrict A_)[N] = A;
  DATA_TYPE (* restrict R_)[N] = R;
  DATA_TYPE (* restrict Q_)[N] = Q;

  const int M_ = _PB_M;
  const int N_ = _PB_N;

#pragma scop
  for (k = 0; k < N_; k++)
  {
    /* 1) Compute the 2-norm of k-th column of A. */
    nrm = SCALAR_VAL(0.0);
    for (i = 0; i < M_; i++)
    {
      const DATA_TYPE aik = A_[i][k];
      nrm += aik * aik;
    }

    const DATA_TYPE rkk = SQRT_FUN(nrm);
    R_[k][k] = rkk;

    /* Precompute reciprocal to turn divisions into multiplications. */
    const DATA_TYPE inv_rkk = SCALAR_VAL(1.0) / rkk;

    /* 2) Compute k-th column of Q: Q[:,k] = A[:,k] / R[k][k]. */
    for (i = 0; i < M_; i++)
    {
      Q_[i][k] = A_[i][k] * inv_rkk;
    }

    /* 3a) Initialize R[k][j] for j > k. */
    for (j = k + 1; j < N_; j++)
      R_[k][j] = SCALAR_VAL(0.0);

    /*
     * 3b) Compute R[k][j] = sum_i Q[i][k] * A[i][j] for all j>k.
     *
     * Original:
     *   for (j = k+1; j < N_; j++) {
     *     R[k][j] = 0;
     *     for (i = 0; i < M_; i++)
     *       R[k][j] += Q[i][k] * A[i][j];
     *   }
     *
     * Transformed:
     *   - For each fixed j, the accumulation over i happens in the same
     *     i-order (0..M_-1), so floating-point rounding behavior is
     *     preserved exactly.
     *   - We traverse A row-wise (i outer, j inner), giving contiguous
     *     accesses in the innermost loop and reusing Q[i][k] across all j.
     */
    for (i = 0; i < M_; i++)
    {
      const DATA_TYPE qik = Q_[i][k];

      /* Operate on row i of A contiguously over j. */
      for (j = k + 1; j < N_; j++)
      {
        R_[k][j] += qik * A_[i][j];
      }
    }

    /*
     * 3c) Update A[:, j] -= Q[:,k] * R[k][j] for all j>k.
     *
     * Original:
     *   for (j = k+1; j < N_; j++)
     *     for (i = 0; i < M_; i++)
     *       A[i][j] = A[i][j] - Q[i][k] * R[k][j];
     *
     * Transformed:
     *   for (i = 0; i < M_; i++) {
     *     qik = Q[i][k];
     *     for (j = k+1; j < N_; j++)
     *       A[i][j] -= qik * R[k][j];
     *   }
     *
     * For each fixed (k,j), the sequence of updates over i is identical,
     * thus preserving numerical results while improving memory locality.
     */
    for (i = 0; i < M_; i++)
    {
      const DATA_TYPE qik = Q_[i][k];

      /* Again, operate on row i contiguously over j. */
      for (j = k + 1; j < N_; j++)
      {
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