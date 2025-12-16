/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization.
 *
 * Optimizations:
 * - Use local restrict pointers for better alias analysis.
 * - Hoist 1/nx and 1/ny divisions out of the inner loop.
 * - Use row pointers inside the i-loop to reduce address arithmetic.
 */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  /* Local restrict-qualified views to help the compiler vectorize. */
  DATA_TYPE (* restrict ex_)[ny]   = ex;
  DATA_TYPE (* restrict ey_)[ny]   = ey;
  DATA_TYPE (* restrict hz_)[ny]   = hz;
  DATA_TYPE * restrict fict_       = _fict_;

  const DATA_TYPE inv_nx = SCALAR_VAL(1.0) / (DATA_TYPE)nx;
  const DATA_TYPE inv_ny = SCALAR_VAL(1.0) / (DATA_TYPE)ny;

  for (i = 0; i < tmax; i++)
    fict_[i] = (DATA_TYPE) i;

  for (i = 0; i < nx; i++)
  {
    DATA_TYPE * restrict ex_i = ex_[i];
    DATA_TYPE * restrict ey_i = ey_[i];
    DATA_TYPE * restrict hz_i = hz_[i];
    const DATA_TYPE di = (DATA_TYPE)i;

    for (j = 0; j < ny; j++)
    {
      const DATA_TYPE dj1 = (DATA_TYPE)(j + 1);
      const DATA_TYPE dj2 = (DATA_TYPE)(j + 2);
      const DATA_TYPE dj3 = (DATA_TYPE)(j + 3);

      ex_i[j] = (di * dj1) * inv_nx;
      ey_i[j] = (di * dj2) * inv_ny;
      hz_i[j] = (di * dj3) * inv_nx;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output.
 *
 * Optimizations:
 * - Use row pointers and restrict-qualified local aliases
 *   to reduce address arithmetic overhead in the inner loop.
 *   (This function is not timed, but we still keep it efficient.)
 */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  DATA_TYPE (* restrict ex_)[ny] = ex;
  DATA_TYPE (* restrict ey_)[ny] = ey;
  DATA_TYPE (* restrict hz_)[ny] = hz;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
  {
    DATA_TYPE * restrict ex_i = ex_[i];
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex_i[j]);
    }
  }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
  {
    DATA_TYPE * restrict ey_i = ey_[i];
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey_i[j]);
    }
  }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
  {
    DATA_TYPE * restrict hz_i = hz_[i];
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz_i[j]);
    }
  }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.
 *
 * Optimizations:
 * - Introduce local restrict-qualified VLA pointers for ex, ey, hz, _fict_
 *   to give the compiler strong no-alias guarantees.
 * - Hoist constant factors (0.5 and 0.7) out of the inner loops.
 * - Use row pointers inside the spatial loops to reduce index
 *   arithmetic and improve cache locality.
 * - Rewrite the ex-update stencil to reuse hz[i][j-1] from a
 *   scalar register (h_prev), reducing memory traffic.
 * - Add GCC vectorization hints (#pragma GCC ivdep) to make sure
 *   the compiler can safely vectorize the innermost loops.
 * - Preserve the original numerical semantics exactly; loop bounds
 *   and update formulas are unchanged.
 */
static
void kernel_fdtd_2d[[gnu::flatten, gnu::noinline]](int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  /* Local restrict-qualified views for better alias analysis.    */
  /* They refer to the same storage as the original parameters.   */
  DATA_TYPE (* restrict ex_)[ny]   = ex;
  DATA_TYPE (* restrict ey_)[ny]   = ey;
  DATA_TYPE (* restrict hz_)[ny]   = hz;
  DATA_TYPE * restrict fict_       = _fict_;

  const int T   = _PB_TMAX;
  const int NX_ = _PB_NX;
  const int NY_ = _PB_NY;

  const DATA_TYPE half          = SCALAR_VAL(0.5);
  const DATA_TYPE seven_tenths  = SCALAR_VAL(0.7);

  int t;

#pragma scop
  for (t = 0; t < T; t++)
  {
    DATA_TYPE fict_t = fict_[t];

    /* 1. Update boundary condition for ey at i = 0:
       ey[0][j] = _fict_[t]; */
    {
      DATA_TYPE * restrict ey0 = ey_[0];
#pragma GCC ivdep
      for (int j = 0; j < NY_; j++)
        ey0[j] = fict_t;
    }

    /* 2. Update ey for i >= 1:
       ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]); */
#pragma GCC ivdep
    for (int i = 1; i < NX_; i++)
    {
      DATA_TYPE * restrict eyi    = ey_[i];
      DATA_TYPE * restrict hzi    = hz_[i];
      DATA_TYPE * restrict hzip1  = hz_[i-1];

#pragma GCC ivdep
      for (int j = 0; j < NY_; j++)
        eyi[j] -= half * (hzi[j] - hzip1[j]);
    }

    /* 3. Update ex for j >= 1:
       ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
       Rewritten to reuse hz[i][j-1] via a scalar h_prev. */
    for (int i = 0; i < NX_; i++)
    {
      DATA_TYPE * restrict exi = ex_[i];
      DATA_TYPE * restrict hzi = hz_[i];

      DATA_TYPE h_prev = hzi[0]; /* hz[i][j-1] for j = 1 */

#pragma GCC ivdep
      for (int j = 1; j < NY_; j++)
      {
        DATA_TYPE h_cur = hzi[j];    /* hz[i][j] */
        exi[j] -= half * (h_cur - h_prev);
        h_prev  = h_cur;
      }
    }

    /* 4. Update hz in the interior:
       hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j]
                                    + ey[i+1][j] - ey[i][j]); */
#pragma GCC ivdep
    for (int i = 0; i < NX_ - 1; i++)
    {
      DATA_TYPE * restrict hzi   = hz_[i];
      DATA_TYPE * restrict exi   = ex_[i];
      DATA_TYPE * restrict eyi   = ey_[i];
      DATA_TYPE * restrict eyip1 = ey_[i+1];

#pragma GCC ivdep
      for (int j = 0; j < NY_ - 1; j++)
      {
        const DATA_TYPE dx = exi[j+1] - exi[j];
        const DATA_TYPE dy = eyip1[j] - eyi[j];
        hzi[j] -= seven_tenths * (dx + dy);
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}