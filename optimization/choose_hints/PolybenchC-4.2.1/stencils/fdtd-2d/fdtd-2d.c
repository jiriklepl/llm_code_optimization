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


/* Array initialization. */
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

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use local `restrict`-qualified pointers to the arrays to help the
     compiler with alias analysis and auto-vectorization.
   - Precompute scalar constants (0.5, 0.7) once per kernel call.
   - Use OpenMP to parallelize the spatial loops inside each time step
     over available cores (ignored if compiled without -fopenmp).
   - Keep inner loops contiguous in memory (j is innermost) and use
     explicit row pointers to reduce index arithmetic.
   - Use `#pragma omp simd` on innermost loops to encourage SIMD
     vectorization where possible.
   These transformations preserve the original numerical semantics.
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
  /* Original loop variables kept for clarity of correspondence
     with the reference implementation; they are not used directly
     in the optimized loops below. */
  int t, i, j;
  (void)t; (void)i; (void)j;

#pragma scop

  /* Effective problem sizes used in the PolyBench loops. */
  const int nx_ = _PB_NX;
  const int ny_ = _PB_NY;
  const int tmax_ = _PB_TMAX;

  /* Create local `restrict` aliases to help the compiler reason about
     aliasing and enable more aggressive vectorization.  The types use
     VLAs so they exactly match the passed-in dimensions. */
  DATA_TYPE (*restrict ex_)[ny]    = ex;
  DATA_TYPE (*restrict ey_)[ny]    = ey;
  DATA_TYPE (*restrict hz_)[ny]    = hz;
  DATA_TYPE *restrict  fict_       = _fict_;

  /* Precompute scalar coefficients. */
  const DATA_TYPE cey = SCALAR_VAL(0.5);
  const DATA_TYPE chz = SCALAR_VAL(0.7);

  /* Parallelize the per-time-step work over space.
     Each time step is executed sequentially (due to temporal
     dependencies), but the spatial loops for a given time step
     are distributed across threads.

     If OpenMP is not enabled at compile time, all `#pragma omp`
     directives are ignored and behavior reduces to a sequential
     optimized kernel identical in semantics to the original. */
#pragma omp parallel
  {
    for (int tt = 0; tt < tmax_; ++tt)
    {
      const DATA_TYPE fict_t = fict_[tt];

      /* Step 1: update the boundary row of ey.
         Independent in j, safe to parallelize. */
#pragma omp for schedule(static)
      for (int jj = 0; jj < ny_; ++jj)
      {
        ey_[0][jj] = fict_t;
      }

      /* Step 2: update ey for rows 1..nx_-1.
         Each (i,j) only touches ey[i][j], hz[i][j], hz[i-1][j];
         no loop-carried dependencies → good for SIMD. */
#pragma omp for schedule(static)
      for (int ii = 1; ii < nx_; ++ii)
      {
        DATA_TYPE *restrict ey_row      = ey_[ii];
        DATA_TYPE *restrict hz_row      = hz_[ii];
        DATA_TYPE *restrict hz_prev_row = hz_[ii-1];

#pragma omp simd
        for (int jj = 0; jj < ny_; ++jj)
        {
          ey_row[jj] -= cey * (hz_row[jj] - hz_prev_row[jj]);
        }
      }

      /* Step 3: update ex for columns 1..ny_-1 in each row.
         Each (i,j) only touches ex[i][j], hz[i][j], hz[i][j-1];
         again, no loop-carried dependencies → good for SIMD. */
#pragma omp for schedule(static)
      for (int ii = 0; ii < nx_; ++ii)
      {
        DATA_TYPE *restrict ex_row = ex_[ii];
        DATA_TYPE *restrict hz_row = hz_[ii];

#pragma omp simd
        for (int jj = 1; jj < ny_; ++jj)
        {
          ex_row[jj] -= cey * (hz_row[jj] - hz_row[jj-1]);
        }
      }

      /* Step 4: update hz on interior points (0..nx_-2, 0..ny_-2).
         Uses the updated ex and ey from the current time step.
         Each (i,j) only reads neighbors in ex,ey and updates hz[i][j]
         → no dependencies across (i,j) iterations. */
#pragma omp for schedule(static)
      for (int ii = 0; ii < nx_ - 1; ++ii)
      {
        DATA_TYPE *restrict hz_row       = hz_[ii];
        DATA_TYPE *restrict ex_row       = ex_[ii];
        DATA_TYPE *restrict ey_row       = ey_[ii];
        DATA_TYPE *restrict ey_next_row  = ey_[ii+1];

#pragma omp simd
        for (int jj = 0; jj < ny_ - 1; ++jj)
        {
          hz_row[jj] -= chz *
                        ( (ex_row[jj+1] - ex_row[jj]) +
                          (ey_next_row[jj] - ey_row[jj]) );
        }
      }

      /* Implicit barrier at the end of each `omp for` ensures that all
         threads complete Steps 1–4 for this time step before any thread
         proceeds to the next time step, preserving temporal dependencies. */
    } /* time loop */
  } /* parallel region */

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