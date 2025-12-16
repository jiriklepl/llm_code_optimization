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

/* --------------------------------------------------------------------
 * Tunable kernel parameters.
 *
 * KERNEL_ADI_I_BLOCK controls the blocking factor for the outer i-loop.
 * It does not change the algorithm, only the order in which independent
 * i-iterations are executed.  Adjust this to better fit the cache
 * hierarchy of a particular machine.
 * ------------------------------------------------------------------ */
#ifndef KERNEL_ADI_I_BLOCK
# define KERNEL_ADI_I_BLOCK 32
#endif


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

  /* Local restrict-qualified aliases for better optimization.
   * The PolyBench macros make the parameters variable-length arrays (VLAs),
   * which decay to pointers to VLA.  Here we create explicit pointer aliases
   * with the same shape but marked restrict, so the compiler can assume
   * that u, v, p, q do not alias and optimize memory accesses more
   * aggressively.
   */
  DATA_TYPE (*restrict u_)[n] = u;
  DATA_TYPE (*restrict v_)[n] = v;
  DATA_TYPE (*restrict p_)[n] = p;
  DATA_TYPE (*restrict q_)[n] = q;

#pragma scop

  /* Frequently used scalar constants. */
  const DATA_TYPE one  = SCALAR_VAL(1.0);
  const DATA_TYPE two  = SCALAR_VAL(2.0);
  const DATA_TYPE zero = SCALAR_VAL(0.0);

  /* Precompute coefficients.  These are identical to the original code,
   * only written more explicitly and with common subexpressions removed.
   */
  DX = one / (DATA_TYPE)_PB_N;
  DY = one / (DATA_TYPE)_PB_N;
  DT = one / (DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  two;
  b =  one + mul1;
  c =  a;
  d = -mul2 / two;
  e =  one + mul2;
  f =  d;

  /* Coefficients (1 + 2*d) and (1 + 2*a) are loop-invariant; precompute
   * once to avoid redundant arithmetic inside the inner loops.
   */
  const DATA_TYPE one_plus_2d = one + two * d;
  const DATA_TYPE one_plus_2a = one + two * a;

  (void)tsteps; /* tsteps is unused; PolyBench uses _PB_TSTEPS instead. */

  for (t = 1; t <= _PB_TSTEPS; t++) {

    /* --------------------------------------------------------------
     * Column sweep: for each column index i, solve a tridiagonal
     * system along the j dimension (forward elimination in p,q and
     * backward substitution in v).
     *
     * We add an i-blocking layer controlled by KERNEL_ADI_I_BLOCK.
     * All iterations over i are independent (u is read-only and
     * p,q,v use disjoint i-slices), so this reordering preserves
     * semantics but allows tuning of cache behavior.
     * ----------------------------------------------------------- */
    for (int ii = 1; ii < _PB_N - 1; ii += KERNEL_ADI_I_BLOCK) {
      int i_end = ii + KERNEL_ADI_I_BLOCK;
      if (i_end > _PB_N - 1)
        i_end = _PB_N - 1;

      for (i = ii; i < i_end; i++) {
        /* Row pointers for p and q at fixed i.  Using row pointers
         * avoids repeated 2D index arithmetic inside the inner loop.
         */
        DATA_TYPE *restrict p_i = p_[i];
        DATA_TYPE *restrict q_i = q_[i];

        /* Boundary conditions at j = 0. */
        v_[0][i] = one;
        p_i[0]   = zero;
        q_i[0]   = v_[0][i];

        /* Pointers that walk down column i-1, i, i+1 of u
         * as j increases.  This makes the column-wise strided
         * access pattern explicit and minimizes address
         * recomputation.
         */
        DATA_TYPE *restrict u_j_im1 = &u_[1][i-1];
        DATA_TYPE *restrict u_j_i   = &u_[1][i];
        DATA_TYPE *restrict u_j_ip1 = &u_[1][i+1];

        DATA_TYPE p_prev = p_i[0]; /* p[i][j-1] */
        DATA_TYPE q_prev = q_i[0]; /* q[i][j-1] */

        for (j = 1; j < _PB_N - 1; j++) {
          /* Common denominator used for both p and q updates. */
          DATA_TYPE denom = a * p_prev + b;

          /* Load u values for this j (three neighboring columns). */
          DATA_TYPE u_im1 = *u_j_im1;
          DATA_TYPE u_i   = *u_j_i;
          DATA_TYPE u_ip1 = *u_j_ip1;

          /* Forward sweep recurrence (unchanged mathematically). */
          DATA_TYPE p_val = -c / denom;
          DATA_TYPE q_val =
              (-d * u_im1 + one_plus_2d * u_i - f * u_ip1 - a * q_prev)
              / denom;

          p_i[j] = p_val;
          q_i[j] = q_val;

          p_prev = p_val;
          q_prev = q_val;

          /* Advance to next j. */
          ++u_j_im1;
          ++u_j_i;
          ++u_j_ip1;
        }

        /* Backward substitution for v along j at fixed i. */
        v_[_PB_N - 1][i] = one;
        DATA_TYPE v_next = v_[_PB_N - 1][i];

        for (j = _PB_N - 2; j >= 1; j--) {
          DATA_TYPE p_val = p_i[j];
          DATA_TYPE q_val = q_i[j];
          DATA_TYPE v_val = p_val * v_next + q_val;
          v_[j][i] = v_val;
          v_next   = v_val;
        }
      }
    }

    /* --------------------------------------------------------------
     * Row sweep: for each row index i, solve a tridiagonal system
     * along the j dimension (forward elimination in p,q and backward
     * substitution in u) using values from v.
     *
     * Again we add a tunable i-blocking layer for cache tuning;
     * iterations over i are independent (v is read-only here).
     * ----------------------------------------------------------- */
    for (int ii = 1; ii < _PB_N - 1; ii += KERNEL_ADI_I_BLOCK) {
      int i_end = ii + KERNEL_ADI_I_BLOCK;
      if (i_end > _PB_N - 1)
        i_end = _PB_N - 1;

      for (i = ii; i < i_end; i++) {
        DATA_TYPE *restrict p_i = p_[i];
        DATA_TYPE *restrict q_i = q_[i];

        /* Boundary conditions at j = 0. */
        u_[i][0] = one;
        p_i[0]   = zero;
        q_i[0]   = u_[i][0];

        /* Pointers that walk along row j (contiguous in memory). */
        DATA_TYPE *restrict v_im1_j = &v_[i-1][1];
        DATA_TYPE *restrict v_i_j   = &v_[i][1];
        DATA_TYPE *restrict v_ip1_j = &v_[i+1][1];

        DATA_TYPE p_prev = p_i[0]; /* p[i][j-1] */
        DATA_TYPE q_prev = q_i[0]; /* q[i][j-1] */

        for (j = 1; j < _PB_N - 1; j++) {
          DATA_TYPE denom = d * p_prev + e;

          DATA_TYPE v_im1 = *v_im1_j;
          DATA_TYPE v_i   = *v_i_j;
          DATA_TYPE v_ip1 = *v_ip1_j;

          DATA_TYPE p_val = -f / denom;
          DATA_TYPE q_val =
              (-a * v_im1 + one_plus_2a * v_i - c * v_ip1 - d * q_prev)
              / denom;

          p_i[j] = p_val;
          q_i[j] = q_val;

          p_prev = p_val;
          q_prev = q_val;

          ++v_im1_j;
          ++v_i_j;
          ++v_ip1_j;
        }

        /* Backward substitution for u along j at fixed i. */
        u_[i][_PB_N - 1] = one;
        DATA_TYPE u_next = u_[i][_PB_N - 1];

        for (j = _PB_N - 2; j >= 1; j--) {
          DATA_TYPE p_val = p_i[j];
          DATA_TYPE q_val = q_i[j];
          DATA_TYPE u_val = p_val * u_next + q_val;
          u_[i][j] = u_val;
          u_next   = u_val;
        }
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