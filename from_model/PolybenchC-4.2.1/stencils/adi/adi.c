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
 *
 * Optimizations applied:
 *  - Hoist and precompute scalar coefficients once.
 *  - Hoist the frequently used stencil centers (1+2*d and 1+2*a).
 *  - Reorganize the column sweep into four phases and interchange loops
 *    so that the tridiagonal recurrences remain along j while the inner
 *    loop is along i (independent), improving cache locality for u and
 *    exposing parallelism/vectorization opportunities.
 *  - Reuse the common denominator in the forward Thomas sweeps to avoid
 *    recomputing a*p[i][j-1]+b and d*p[i][j-1]+e.
 *  - Rewrite the backward column sweep using an increasing index jb
 *    instead of a decreasing j loop to keep affine loop bounds.
 *
 * All algebraic formulas are preserved; only loop ordering of
 * independent iterations and local common subexpressions have changed.
 */
static
void kernel_adi[[gnu::flatten, gnu::noinline]](int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  (void)tsteps; /* tsteps is represented by _PB_TSTEPS; silence unused warning if any. */
  (void)n;      /* n is represented by _PB_N; silence unused warning if any. */

  int t, i, j, jb;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;
  /* Precomputed centers for the stencils: 1 + 2*d and 1 + 2*a. */
  DATA_TYPE alpha_col, alpha_row;

#pragma scop

  /* Coefficient setup (identical algebra to the original code). */
  DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0) + mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0) + mul2;
  f = d;

  /* Hoisted constants that are reused in the inner loops. */
  alpha_col = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d; /* 1 + 2*d */
  alpha_row = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a; /* 1 + 2*a */

  /* Convenience: last interior index + 1 (i, j run from 1 to _PB_N-2). */
  const int n_inner = _PB_N - 1;

  for (t = 1; t <= _PB_TSTEPS; t++)
  {
    /* =========================
       Column sweep (implicit in j)
       ========================= */

    /* 1) Column boundary conditions and initialization for forward sweep.
       Each interior column i defines an independent 1D tridiagonal system. */
    for (i = 1; i < n_inner; i++)
    {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i]; /* = 1.0 */
    }

    /* 2) Forward sweep along j (Thomas factorization).
          - j is the recurrence dimension and remains sequential.
          - For fixed j, all i-lines are independent. */
    for (j = 1; j < n_inner; j++)
    {
      for (i = 1; i < n_inner; i++)
      {
        /* Common denominator: a * p[i][j-1] + b.
           It is used by both p[i][j] and q[i][j]. */
        DATA_TYPE denom = a * p[i][j-1] + b;

        p[i][j] = -c / denom;

        q[i][j] =
          (-d * u[j][i-1] +
           alpha_col * u[j][i] -
           f * u[j][i+1] -
           a * q[i][j-1]) / denom;
      }
    }

    /* 3) Column boundary at j = N-1 for backward substitution. */
    for (i = 1; i < n_inner; i++)
    {
      v[_PB_N-1][i] = SCALAR_VAL(1.0);
    }

    /* 4) Backward sweep in j to update v using p and q.
          Original loop: for (j = N-2; j >= 1; j--)
            v[j][i] = p[i][j]*v[j+1][i] + q[i][j];
          Here we express it with an increasing jb index to keep
          affine bounds while preserving the descending order of j. */
    for (jb = 1; jb < n_inner; jb++)
    {
      int j_idx = _PB_N - 1 - jb; /* j_idx : N-2, N-3, ..., 1 */
      for (i = 1; i < n_inner; i++)
      {
        v[j_idx][i] = p[i][j_idx] * v[j_idx+1][i] + q[i][j_idx];
      }
    }

    /* ======================
       Row sweep (implicit in i)
       ====================== */

    /* 5) Row boundary conditions and initialization for forward sweep. */
    for (i = 1; i < n_inner; i++)
    {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0]; /* = 1.0 */
    }

    /* 6) Forward sweep along j for each row (Thomas factorization).
          Loop ordering (i outer, j inner) matches row-major storage and
          is kept as in the original code; only the denominator is reused. */
    for (i = 1; i < n_inner; i++)
    {
      for (j = 1; j < n_inner; j++)
      {
        DATA_TYPE denom = d * p[i][j-1] + e;

        p[i][j] = -f / denom;

        q[i][j] =
          (-a * v[i-1][j] +
           alpha_row * v[i][j] -
           c * v[i+1][j] -
           d * q[i][j-1]) / denom;
      }

      /* 7) Row boundary at j = N-1 for backward substitution. */
      u[i][_PB_N-1] = SCALAR_VAL(1.0);

      /* 8) Backward sweep in j to update u using p and q.
            Original descending loop is preserved here. */
      for (j = _PB_N-2; j >= 1; j--)
      {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
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
  kernel_adi (tsteps, n,
              POLYBENCH_ARRAY(u),
              POLYBENCH_ARRAY(v),
              POLYBENCH_ARRAY(p),
              POLYBENCH_ARRAY(q));

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