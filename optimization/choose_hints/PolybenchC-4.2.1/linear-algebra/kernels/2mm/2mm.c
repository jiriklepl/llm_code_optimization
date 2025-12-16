/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimized version:
   - Reorders the matrix multiplications to use an (i,k,j) loop
     ordering, which is more cache-friendly for row-major storage:
       * A accessed by rows (A[i][k])
       * B accessed by rows (B[k][j])
       * tmp and D accessed by rows (tmp[i][j], D[i][j])
   - Splits the original kernel into four phases:
       1) tmp = 0
       2) tmp = alpha * A * B
       3) D   = beta * D
       4) D  += tmp * C
     This preserves the mathematical expression
       D := alpha * A * B * C + beta * D
     and, for every (i,j), keeps the accumulation over k in the same
     order as the original code (k from 0 to N-1), so the
     floating‑point reduction order is unchanged.
   - The outer loops over i are annotated with OpenMP 'parallel for'.
     These pragmas are ignored unless the code is compiled with
     -fopenmp, in which case the work is distributed across threads.
 */
static
void kernel_2mm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  /* PolyBench loop bounds (typically compile-time constants). */
  const int ni_ = _PB_NI;
  const int nj_ = _PB_NJ;
  const int nk_ = _PB_NK;
  const int nl_ = _PB_NL;

  const DATA_TYPE alpha_ = alpha;
  const DATA_TYPE beta_  = beta;

  int i, j, k;

#pragma scop
  /* 1. Initialize tmp to zero.
     Separated from the accumulation loop so that we can use an
     outer‑product formulation for the matrix multiplication. */
#pragma omp parallel for schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE *tmp_row = tmp[i];
    for (j = 0; j < nj_; j++)
      tmp_row[j] = SCALAR_VAL(0.0);
  }

  /* 2. tmp := alpha * A * B
     Outer‑product style with loop order (i,k,j) for better locality:

       for i
         for k
           a_ik = alpha * A[i][k];
           for j
             tmp[i][j] += a_ik * B[k][j];

     This touches A, B and tmp in contiguous memory along the inner
     loop, and for each (i,j) the contributions are accumulated over k
     in strictly increasing order, as in the original code. */
#pragma omp parallel for schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE *tmp_row = tmp[i];
    DATA_TYPE *A_row   = A[i];

    for (k = 0; k < nk_; k++)
    {
      const DATA_TYPE a_ik = alpha_ * A_row[k];
      DATA_TYPE *B_row = B[k];

      for (j = 0; j < nj_; j++)
        tmp_row[j] += a_ik * B_row[j];
    }
  }

  /* 3. Scale D by beta: D := beta * D. */
#pragma omp parallel for schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE *D_row = D[i];
    for (j = 0; j < nl_; j++)
      D_row[j] *= beta_;
  }

  /* 4. D += tmp * C
       D(i,j) = beta * D(i,j) + sum_k tmp(i,k) * C(k,j)

     Again we use (i,k,j) ordering, updating whole rows at a time:

       for i
         for k
           t = tmp[i][k];
           for j
             D[i][j] += t * C[k][j];

     This keeps accesses to tmp, C and D row‑major and preserves the
     original accumulation order over k for each (i,j). */
#pragma omp parallel for schedule(static)
  for (i = 0; i < ni_; i++)
  {
    DATA_TYPE *D_row   = D[i];
    DATA_TYPE *tmp_row = tmp[i];

    for (k = 0; k < nj_; k++)
    {
      const DATA_TYPE tmp_ik = tmp_row[k];
      DATA_TYPE *C_row = C[k];

      for (j = 0; j < nl_; j++)
        D_row[j] += tmp_ik * C_row[j];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
	      alpha, beta,
	      POLYBENCH_ARRAY(tmp),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}