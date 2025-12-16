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
  /* Use a restrict-qualified alias to help the compiler with
     alias analysis and enable better vectorization. */
  DATA_TYPE (* restrict u_)[n] = u;

  /* Precompute 1/n once to avoid a division in the inner loop.
     This is algebraically equivalent to dividing by n for each
     element and significantly cheaper. */
  const DATA_TYPE inv_n = SCALAR_VAL(1.0) / (DATA_TYPE)n;

  /* Parallelize initialization across rows when OpenMP is enabled.
     Without -fopenmp, this pragma is ignored and the code is
     executed sequentially, preserving original behavior. */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
  {
    DATA_TYPE * restrict u_i = u_[i];
    const DATA_TYPE base = (DATA_TYPE)i + (DATA_TYPE)n;

    for (int j = 0; j < n; j++)
    {
      /* Original formula: u[i][j] = (i + n - j) / n */
      u_i[j] = (base - (DATA_TYPE)j) * inv_n;
    }
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
  /* Use local restrict-qualified aliases; this tells the compiler that
     these 2D arrays do not alias each other, which enables more
     aggressive optimization and better use of registers. */
  DATA_TYPE (* restrict u_)[n] = u;
  DATA_TYPE (* restrict v_)[n] = v;
  DATA_TYPE (* restrict p_)[n] = p;
  DATA_TYPE (* restrict q_)[n] = q;

  const int n_eff = _PB_N;

  /* Scalar precomputations (identical numerically to the original code). */
  const DATA_TYPE one  = SCALAR_VAL(1.0);
  const DATA_TYPE two  = SCALAR_VAL(2.0);
  const DATA_TYPE zero = SCALAR_VAL(0.0);

  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = one / (DATA_TYPE)n_eff;
  DY = one / (DATA_TYPE)n_eff;
  DT = one / (DATA_TYPE)_PB_TSTEPS;

  B1 = SCALAR_VAL(2.0);
  B2 = one;

  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 / two;
  b = one + mul1;
  c = a;
  d = -mul2 / two;
  e = one + mul2;
  f = d;

  /* These combinations are used repeatedly inside the innermost loops. */
  const DATA_TYPE one_plus_2d    = one + two * d;
  const DATA_TYPE one_plus_2a    = one + two * a;
  const DATA_TYPE boundary_value = one;

  /* Parallelize over the independent i-dimension (rows/columns).
     Each tridiagonal solve in j is inherently sequential, but the
     solves for different i are independent and can be executed in
     parallel.  When OpenMP is not enabled, the pragmas are ignored
     and execution is purely sequential (original behavior). */
#pragma omp parallel
  {
    for (int t = 1; t <= _PB_TSTEPS; ++t)
    {
      /* Column sweep: for each column index i, solve along j. */
#pragma omp for schedule(static)
      for (int i = 1; i < n_eff-1; ++i)
      {
        DATA_TYPE * restrict p_i = p_[i];
        DATA_TYPE * restrict q_i = q_[i];

        v_[0][i] = boundary_value;
        p_i[0]   = zero;
        q_i[0]   = boundary_value;

        /* Forward sweep (Thomas algorithm) in the j direction. */
        for (int j = 1; j < n_eff-1; ++j)
        {
          const DATA_TYPE u_j_im1 = u_[j][i-1];
          const DATA_TYPE u_j_i   = u_[j][i];
          const DATA_TYPE u_j_ip1 = u_[j][i+1];

          const DATA_TYPE denom     = a * p_i[j-1] + b;
          const DATA_TYPE inv_denom = one / denom; /* Algebraically equivalent to division. */
          const DATA_TYPE rhs       = -d * u_j_im1 + one_plus_2d * u_j_i
                                     - f * u_j_ip1 - a * q_i[j-1];

          /* p_i[j] = -c / denom; q_i[j] = rhs / denom; */
          p_i[j] = -c * inv_denom;
          q_i[j] = rhs * inv_denom;
        }

        v_[n_eff-1][i] = boundary_value;

        /* Backward substitution in j. */
        for (int j = n_eff-2; j >= 1; --j)
        {
          v_[j][i] = p_i[j] * v_[j+1][i] + q_i[j];
        }
      }

      /* Row sweep: for each row index i, solve along j. */
#pragma omp for schedule(static)
      for (int i = 1; i < n_eff-1; ++i)
      {
        DATA_TYPE * restrict p_i = p_[i];
        DATA_TYPE * restrict q_i = q_[i];
        DATA_TYPE * restrict u_i = u_[i];

        u_i[0] = boundary_value;
        p_i[0] = zero;
        q_i[0] = boundary_value;

        /* Forward sweep in j using the intermediate array v_. */
        for (int j = 1; j < n_eff-1; ++j)
        {
          const DATA_TYPE v_im1_j = v_[i-1][j];
          const DATA_TYPE v_i_j   = v_[i][j];
          const DATA_TYPE v_ip1_j = v_[i+1][j];

          const DATA_TYPE denom     = d * p_i[j-1] + e;
          const DATA_TYPE inv_denom = one / denom; /* Algebraically equivalent to division. */
          const DATA_TYPE rhs       = -a * v_im1_j + one_plus_2a * v_i_j
                                     - c * v_ip1_j - d * q_i[j-1];

          /* p_i[j] = -f / denom; q_i[j] = rhs / denom; */
          p_i[j] = -f * inv_denom;
          q_i[j] = rhs * inv_denom;
        }

        u_i[n_eff-1] = boundary_value;

        /* Backward substitution in j. */
        for (int j = n_eff-2; j >= 1; --j)
        {
          u_i[j] = p_i[j] * u_i[j+1] + q_i[j];
        }
      }
    } /* end of time-stepping loop */
  }   /* end of OpenMP parallel region */

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
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v),
              POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

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