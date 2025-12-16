/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
 
static 
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n)) 

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi[[gnu::flatten, gnu::noinline]](int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  /* Effective problem sizes (PolyBench may redefine _PB_* macros). */
  const int nPB  = _PB_N;
  const int tPB  = _PB_TSTEPS;

  /* Constant scalars used throughout the kernel.  Hoist SCALAR_VAL out
     of the inner loops to reduce redundant work. */
  const DATA_TYPE one  = SCALAR_VAL(1.0);
  const DATA_TYPE two  = SCALAR_VAL(2.0);
  const DATA_TYPE zero = SCALAR_VAL(0.0);

  /* Use local restrict-qualified pointers to help the optimizer:
     u_, v_, p_, q_ are the only ways we access the underlying data,
     and they are guaranteed not to alias each other. */
  DATA_TYPE (*restrict u_)[n] = (DATA_TYPE (*)[n])u;
  DATA_TYPE (*restrict v_)[n] = (DATA_TYPE (*)[n])v;
  DATA_TYPE (*restrict p_)[n] = (DATA_TYPE (*)[n])p;
  DATA_TYPE (*restrict q_)[n] = (DATA_TYPE (*)[n])q;

  /* Precompute coefficients (unchanged numerically). */
  DX  = one / (DATA_TYPE)nPB;
  DY  = one / (DATA_TYPE)nPB;
  DT  = one / (DATA_TYPE)tPB;
  B1  = SCALAR_VAL(2.0);
  B2  = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 / two;
  b = one + mul1;
  c = a;
  d = -mul2 / two;
  e = one + mul2;
  f = d;

  /* Diagonal coefficients that appear repeatedly in the inner loops. */
  const DATA_TYPE col_center = one + two * d; /* (1 + 2*d) in column sweep */
  const DATA_TYPE row_center = one + two * a; /* (1 + 2*a) in row sweep   */

  for (t = 1; t <= tPB; t++) {
    /* -----------------------------------------------------------------
     * Column sweep
     * Each column i is independent given u, so we parallelize over i.
     * The recurrences along j are kept sequential inside each thread.
     * ----------------------------------------------------------------- */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i,j) \
  shared(u_, v_, p_, q_, nPB, a, b, c, d, f, one, zero, col_center)
#endif
    for (i = 1; i < nPB-1; i++) {
      DATA_TYPE * restrict p_i = p_[i];
      DATA_TYPE * restrict q_i = q_[i];

      /* Boundary conditions. */
      v_[0][i]   = one;
      p_i[0]     = zero;
      q_i[0]     = one; /* equals v_[0][i] */

      /* Forward tridiagonal solve along column i. */
      for (j = 1; j < nPB-1; j++) {
        DATA_TYPE * restrict u_j = u_[j];

        const DATA_TYPE p_prev   = p_i[j-1];
        const DATA_TYPE q_prev   = q_i[j-1];

        /* Common denominator used for both p_i[j] and q_i[j].
           Compute its reciprocal once to avoid two divisions. */
        const DATA_TYPE denom     = a * p_prev + b;
        const DATA_TYPE inv_denom = one / denom;

        p_i[j] = (-c) * inv_denom;

        const DATA_TYPE num_q =
          (-d * u_j[i-1]) +
          (col_center * u_j[i]) -
          (f * u_j[i+1]) -
          (a * q_prev);

        q_i[j] = num_q * inv_denom;
      }

      /* Backward substitution to obtain v along column i. */
      v_[nPB-1][i] = one;
      for (j = nPB-2; j >= 1; j--) {
        v_[j][i] = p_i[j] * v_[j+1][i] + q_i[j];
      }
    }

    /* -----------------------------------------------------------------
     * Row sweep
     * Symmetric to column sweep: each row i is independent given v,
     * so we parallelize over i; j remains sequential.
     * ----------------------------------------------------------------- */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i,j) \
  shared(u_, v_, p_, q_, nPB, a, c, d, e, f, one, zero, row_center)
#endif
    for (i = 1; i < nPB-1; i++) {
      DATA_TYPE * restrict p_i = p_[i];
      DATA_TYPE * restrict q_i = q_[i];
      DATA_TYPE * restrict u_i = u_[i];

      DATA_TYPE * restrict v_im1 = v_[i-1];
      DATA_TYPE * restrict v_i   = v_[i];
      DATA_TYPE * restrict v_ip1 = v_[i+1];

      /* Boundary conditions. */
      u_i[0] = one;
      p_i[0] = zero;
      q_i[0] = one; /* equals u_i[0] */

      /* Forward tridiagonal solve along row i. */
      for (j = 1; j < nPB-1; j++) {
        const DATA_TYPE p_prev   = p_i[j-1];
        const DATA_TYPE q_prev   = q_i[j-1];

        const DATA_TYPE denom     = d * p_prev + e;
        const DATA_TYPE inv_denom = one / denom;

        p_i[j] = (-f) * inv_denom;

        const DATA_TYPE num_q =
          (-a * v_im1[j]) +
          (row_center * v_i[j]) -
          (c * v_ip1[j]) -
          (d * q_prev);

        q_i[j] = num_q * inv_denom;
      }

      /* Backward substitution to obtain u along row i. */
      u_i[nPB-1] = one;
      for (j = nPB-2; j >= 1; j--) {
        u_i[j] = p_i[j] * u_i[j+1] + q_i[j];
      }
    }
  }

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}