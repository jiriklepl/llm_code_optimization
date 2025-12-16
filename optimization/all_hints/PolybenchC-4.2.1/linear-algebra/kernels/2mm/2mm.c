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

/* Optional: only included when compiling with OpenMP support. */
#ifdef _OPENMP
# include <omp.h>
#endif

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

   Optimizations:
   - Reordered loops to improve spatial locality:
       * First GEMM (tmp = alpha * A * B):  i-k-j order
         gives contiguous accesses in j for both B and tmp.
       * Second GEMM (D += tmp * C):       i-k-j order
         gives contiguous accesses in j for both C and D.
   - Separated initialization of tmp from the GEMM to allow better
     vectorization and parallelization.
   - Added an OpenMP parallel region that parallelizes all three
     phases over the i dimension. If compiled without OpenMP
     support, the pragmas are ignored and the code remains serial.
   - Used GCC's ivdep pragma on innermost loops to help vectorization.
   - The order of floating-point operations for each individual
     element of tmp and D with respect to the reduction index k
     is preserved (k is still traversed in increasing order),
     so numerical behavior remains consistent with the original code.
*/
static
void kernel_2mm[[gnu::flatten, gnu::noinline]](int ni, int nj, int nk, int nl,
		const DATA_TYPE alpha,
		const DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  /* Keep original variables; thread-private copies are declared
     inside the OpenMP region. */
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */

  /* Single parallel region to amortize thread creation/destruction
     over the whole kernel. The work is split over the i dimension,
     which is independent across rows. */
#pragma omp parallel
  {
    int i, j, k;

    /* 1) Initialize tmp to 0.0
       Separated from the GEMM to allow a clean, vectorizable loop
       and parallelization across i. */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
        tmp[i][j] = SCALAR_VAL(0.0);

    /* 2) First matrix multiplication:
           tmp = alpha * A * B

       Original code (per element):
         tmp[i][j] = 0;
         for k: tmp[i][j] += alpha * A[i][k] * B[k][j];

       Here we use the more cache-friendly i-k-j order:
         for i:
           for k:
             aik = alpha * A[i][k];
             for j:
               tmp[i][j] += aik * B[k][j];

       For any fixed (i,j), the contributions are still added in
       strictly increasing k order, preserving the original
       accumulation sequence for that element. */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NI; i++) {
      for (k = 0; k < _PB_NK; k++) {
        DATA_TYPE aik = alpha * A[i][k];

        /* No loop-carried dependencies across j for this row. */
#pragma GCC ivdep
        for (j = 0; j < _PB_NJ; j++) {
          tmp[i][j] += aik * B[k][j];
        }
      }
    }

    /* 3) Second matrix multiplication and scaling:
           D = beta * D + tmp * C

       Original code (per element):
         D[i][j] *= beta;
         for k: D[i][j] += tmp[i][k] * C[k][j];

       We restructure to:
         for j: D[i][j] *= beta;
         for k:
           t = tmp[i][k];
           for j: D[i][j] += t * C[k][j];

       For each (i,j), the scaling by beta still happens before
       any additions, and the additions over k still occur in
       increasing k order. */
#pragma omp for schedule(static)
    for (i = 0; i < _PB_NI; i++) {

      /* Scale row i of D by beta. */
#pragma GCC ivdep
      for (j = 0; j < _PB_NL; j++) {
        D[i][j] *= beta;
      }

      /* Add tmp * C into D. Loop order improves locality for both
         C[k][j] and D[i][j] by iterating over contiguous j. */
      for (k = 0; k < _PB_NJ; k++) {
        DATA_TYPE tmp_ik = tmp[i][k];

#pragma GCC ivdep
        for (j = 0; j < _PB_NL; j++) {
          D[i][j] += tmp_ik * C[k][j];
        }
      }
    }
  } /* end OpenMP parallel region */

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