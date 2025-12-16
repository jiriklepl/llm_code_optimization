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
  /* Loop indices are kept as in the original code, but we will
   * explicitly control their data-sharing properties in OpenMP. */
  int t, i, j;

  /* Local pointer aliases with 'restrict' to give the compiler
   * stronger aliasing guarantees. The PolyBench harness always
   * passes distinct, contiguous arrays, so this is safe and
   * improves optimization opportunities. */
  DATA_TYPE (*restrict u_)[n] = u;
  DATA_TYPE (*restrict v_)[n] = v;
  DATA_TYPE (*restrict p_)[n] = p;
  DATA_TYPE (*restrict q_)[n] = q;

  /* Scalar coefficients. Mark them const so the compiler can
   * freely propagate and keep them in registers. */
  const DATA_TYPE DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  const DATA_TYPE DY = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  const DATA_TYPE DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
  const DATA_TYPE B1 = SCALAR_VAL(2.0);
  const DATA_TYPE B2 = SCALAR_VAL(1.0);
  const DATA_TYPE mul1 = B1 * DT / (DX * DX);
  const DATA_TYPE mul2 = B2 * DT / (DY * DY);

  const DATA_TYPE a = -mul1 / SCALAR_VAL(2.0);
  const DATA_TYPE b = SCALAR_VAL(1.0) + mul1;
  const DATA_TYPE c = a;
  const DATA_TYPE d = -mul2 / SCALAR_VAL(2.0);
  const DATA_TYPE e = SCALAR_VAL(1.0) + mul2;
  const DATA_TYPE f = d;

  /* Frequently used scalar combinations, precomputed once. */
  const DATA_TYPE one = SCALAR_VAL(1.0);
  const DATA_TYPE two = SCALAR_VAL(2.0);
  const DATA_TYPE one_plus_2d = one + two * d;
  const DATA_TYPE one_plus_2a = one + two * a;

#pragma scop

  /* Parallelization strategy:
   *  - Each time step 't' is fundamentally sequential (data dependence),
   *    so we cannot parallelize over 't'.
   *  - For a fixed 't', each line/column (index 'i') is independent in
   *    both the column sweep and the row sweep, so we parallelize over 'i'.
   *  - The inner recurrences in 'j' keep their original order.
   *
   * We use a single OpenMP parallel region around the whole time loop
   * to amortize thread creation/destruction overhead.
   *
   * If compiled without OpenMP support, the pragmas are ignored and the
   * code executes sequentially with identical semantics.
   */
#pragma omp parallel private(t,i,j)
  {
    for (t = 1; t <= _PB_TSTEPS; t++)
    {
      /* ---------------------------------------------------------
       * Column sweep
       * --------------------------------------------------------- */
#pragma omp for schedule(static)
      for (i = 1; i < _PB_N - 1; i++)
      {
        /* p_[i][*] and q_[i][*] are used as 1D contiguous rows. */
        DATA_TYPE *restrict p_i = p_[i];
        DATA_TYPE *restrict q_i = q_[i];

        v_[0][i] = one;
        p_i[0]   = SCALAR_VAL(0.0);
        q_i[0]   = v_[0][i];

        /* Forward sweep: tridiagonal solve along the j dimension.
         * The recurrence in j is preserved. We reduce the number of
         * divisions by computing 1/den once per j and reusing it for
         * both p and q. */
        for (j = 1; j < _PB_N - 1; j++)
        {
          /* Access the current "row" j of u_ only once. */
          DATA_TYPE *restrict u_j = u_[j];

          DATA_TYPE den    = a * p_i[j - 1] + b;
          DATA_TYPE invden = one / den;

          DATA_TYPE num =
            (-d) * u_j[i - 1] +
            one_plus_2d * u_j[i] +
            (-f) * u_j[i + 1] -
            a * q_i[j - 1];

          p_i[j] = (-c) * invden;
          q_i[j] = num * invden;
        }

        /* Backward substitution along j. */
        v_[_PB_N - 1][i] = one;
        for (j = _PB_N - 2; j >= 1; j--)
        {
          v_[j][i] = p_i[j] * v_[j + 1][i] + q_i[j];
        }
      }

      /* ---------------------------------------------------------
       * Row sweep
       * --------------------------------------------------------- */
#pragma omp for schedule(static)
      for (i = 1; i < _PB_N - 1; i++)
      {
        /* Row pointers for better locality on the inner j loop. */
        DATA_TYPE *restrict u_i = u_[i];
        DATA_TYPE *restrict p_i = p_[i];
        DATA_TYPE *restrict q_i = q_[i];

        u_i[0] = one;
        p_i[0] = SCALAR_VAL(0.0);
        q_i[0] = u_i[0];

        /* Forward sweep: tridiagonal solve along the j dimension. */
        for (j = 1; j < _PB_N - 1; j++)
        {
          DATA_TYPE den    = d * p_i[j - 1] + e;
          DATA_TYPE invden = one / den;

          DATA_TYPE num =
            (-a) * v_[i - 1][j] +
            one_plus_2a * v_[i][j] +
            (-c) * v_[i + 1][j] -
            d * q_i[j - 1];

          p_i[j] = (-f) * invden;
          q_i[j] = num * invden;
        }

        /* Backward substitution along j. */
        u_i[_PB_N - 1] = one;
        for (j = _PB_N - 2; j >= 1; j--)
        {
          u_i[j] = p_i[j] * u_i[j + 1] + q_i[j];
        }
      }
      /* Implicit barriers at the end of each 'omp for' keep the
       * column and row sweeps properly ordered across all threads. */
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