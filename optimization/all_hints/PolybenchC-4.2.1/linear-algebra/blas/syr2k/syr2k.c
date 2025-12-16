/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
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
void kernel_syr2k[[gnu::flatten, gnu::noinline]](int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  /* Use local restrict-qualified aliases to help the compiler with
     alias analysis and enable more aggressive vectorization.
     The POLYBENCH_2D macro expands to variable-length array (VLA)
     types, so we mirror that layout here. */
  DATA_TYPE (*restrict C_)[n] = (DATA_TYPE (*restrict)[n]) C;
  DATA_TYPE (*restrict A_)[m] = (DATA_TYPE (*restrict)[m]) A;
  DATA_TYPE (*restrict B_)[m] = (DATA_TYPE (*restrict)[m]) B;

  const int n_ = _PB_N;
  const int m_ = _PB_M;

  const DATA_TYPE alpha_ = alpha;
  const DATA_TYPE beta_  = beta;

  int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  /* Optimized loop nest:
     - Original order:
         for i
           for j <= i   C[i][j] *= beta
           for k
             for j <= i C[i][j] += ...

       This updated C[i][j] in memory on every (i,j,k) iteration.

     - New order:
         for i
           for j <= i
             cij = C[i][j] * beta
             for k
               cij += ...
             C[i][j] = cij

       Now each C(i,j) element is:
         * loaded once
         * kept in a register across the whole k-reduction
         * stored once

       This greatly reduces memory traffic to C and exposes a
       simple, contiguous inner loop over k for vectorization.
  */
#ifdef _OPENMP
  /* Parallelize outer loop over rows of C when OpenMP is available.
     Each (i,j) pair is independent, so there are no data races. */
#pragma omp parallel for schedule(static) private(j,k) \
    shared(C_,A_,B_,alpha_,beta_,n_,m_)
#endif
  for (i = 0; i < n_; i++) {
    DATA_TYPE *Ci = C_[i];
    DATA_TYPE *Ai = A_[i];
    DATA_TYPE *Bi = B_[i];

    for (j = 0; j <= i; j++) {
      /* Scale once by beta, as in the original code
         (which did C[i][j] *= beta before the k-loop). */
      DATA_TYPE cij = Ci[j] * beta_;

      DATA_TYPE *Aj = A_[j];
      DATA_TYPE *Bj = B_[j];

      /* Innermost k-loop: pure reduction into cij.
         - Accesses Aj[k], Bj[k], Ai[k], Bi[k] all with unit stride,
           which improves cache and TLB behavior compared to the
           original j-inner ordering.
         - No loop-carried dependencies on cij besides the reduction,
           so the compiler is free to vectorize.
      */
#pragma GCC ivdep
      for (k = 0; k < m_; k++) {
        /* Algebraically equivalent transformation of the original:
             original: C[i][j] += A[j][k]*alpha*B[i][k]
                                    + B[j][k]*alpha*A[i][k];
           Factor out alpha to reduce one multiplication per k:
             C[i][j] += alpha * (A[j][k]*B[i][k] + B[j][k]*A[i][k]);
        */
        cij += alpha_ * (Aj[k] * Bi[k] + Bj[k] * Ai[k]);
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
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syr2k (n, m,
		alpha, beta,
		POLYBENCH_ARRAY(C),
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}