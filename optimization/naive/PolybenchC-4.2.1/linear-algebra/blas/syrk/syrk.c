/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   BLAS PARAMS
   -----------
   TRANS = 'N'
   UPLO  = 'L'
   => Form  C := alpha*A*A**T + beta*C.
   A is NxM
   C is NxN
*/
static
void kernel_syrk[[gnu::flatten, gnu::noinline]](int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  /* Local copies of the problem sizes (kept constant throughout the kernel). */
  const int n_local = n;
  const int m_local = m;

  /* Row stride for A in elements. Since A is laid out as A[n][m],
     two consecutive rows differ by exactly 'm_local' elements. */
  const int strideA = m_local;

#pragma scop
  for (int i = 0; i < n_local; ++i)
  {
    /* Pointers to the i-th row of C and A.
       Mark them 'restrict' to enable better vectorization and
       to tell the compiler that, within this iteration, no other
       pointer aliases these rows. */
    DATA_TYPE       *restrict Ci = C[i];
    const DATA_TYPE *restrict Ai = A[i];

    /* 1) Scale the lower-triangular part of row i of C by beta.
          C[i][j] = beta * C[i][j],  for 0 <= j <= i.
       Accesses to C are contiguous in memory, which is SIMD- and
       cache-friendly. */
    for (int j = 0; j <= i; ++j)
      Ci[j] *= beta;

    /* 2) Symmetric rank-k update:
          C[i][j] += alpha * A[i][k] * A[j][k],
          for 0 <= j <= i, 0 <= k < m_local.

       We keep the original (i, k, j) loop ordering to preserve the
       accumulation order for each C[i][j], but:

       - 'alpha * A[i][k]' is hoisted out of the inner j-loop to avoid
         recomputing it (reduces multiplications by ~ (i+1) per k).
       - We traverse column k of A with an explicit pointer that advances
         by 'strideA' elements, avoiding repeated 2D index arithmetic. */
    for (int k = 0; k < m_local; ++k)
    {
      const DATA_TYPE aik = alpha * Ai[k];  /* Reused for all j. */

      /* Pointer to the first element of column k: A[0][k]. */
      const DATA_TYPE *Aj = &A[0][k];

      /* Pointer walking along C[i][0..i]. */
      DATA_TYPE *Cij = Ci;

      /* Inner loop over the lower-triangular part of row i. */
      for (int j = 0; j <= i; ++j)
      {
        /* Original update:
             C[i][j] += alpha * A[i][k] * A[j][k];
           implemented as:
             *Cij      += aik * (*Aj);
         */
        *Cij += aik * (*Aj);

        /* Move to C[i][j+1] and A[j+1][k]. */
        ++Cij;
        Aj += strideA;
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}