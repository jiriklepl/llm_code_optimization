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
   including the call and return. */
static
void kernel_syrk[[gnu::flatten, gnu::noinline]](int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

  /* Create local restrict-qualified views of the 2D arrays.
     This helps the compiler with alias analysis while keeping the
     original function interface (required by the PolyBench harness). */
  DATA_TYPE (* __restrict__ Crestrict)[n] = C;
  DATA_TYPE (* __restrict__ Arestrict)[m] = A;

  const DATA_TYPE alpha_val = alpha;
  const DATA_TYPE beta_val  = beta;

  // BLAS PARAMS
  // TRANS = 'N'
  // UPLO  = 'L'
  // =>  Form  C := alpha*A*A**T + beta*C.
  // A is NxM
  // C is NxN

#pragma scop
  /* Parallelize the outer i-loop.
     - Each iteration i writes only to row i of the lower triangle (j <= i),
       so different iterations do not race on C.
     - A is read-only inside the kernel.
     If OpenMP is not enabled at compile time, this pragma is ignored. */
#pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < _PB_N; i++) {

    /* Row pointers improve locality and reduce index arithmetic. */
    DATA_TYPE * __restrict__ Ci = Crestrict[i];
    DATA_TYPE * __restrict__ Ai = Arestrict[i];

    /* Compute the lower-triangular part of row i. */
    for (j = 0; j <= i; j++) {

      DATA_TYPE * __restrict__ Aj = Arestrict[j];

      /* Use a local accumulator kept in a register.
         This improves data locality (single load/store of C[i][j])
         and exposes a simple inner dot-product over k that the
         compiler can efficiently vectorize (contiguous access in k). */
      DATA_TYPE cij = Ci[j] * beta_val;

      /* Inner product of rows i and j of A.
         Loop order (i, j, k) is chosen so that, for each pair (i,j),
         the k-loop walks contiguous memory in both Ai and Aj
         (row-major storage), which significantly improves cache
         behavior and SIMD vectorization compared to the original
         (i, k, j) ordering.

         The sequence of floating-point operations for each element
         C[i][j] is preserved with respect to the original code:
         - First scale C[i][j] by beta,
         - Then accumulate alpha * A[i][k] * A[j][k] for k=0..M-1,
         in the same k-order as before. Only the interleaving of
         operations for different (i,j) pairs changes, which does not
         affect the values because those updates are independent. */
      for (k = 0; k < _PB_M; k++) {
        cij += alpha_val * Ai[k] * Aj[k];
      }

      Ci[j] = cij;
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