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
#include <omp.h>  /* Added for OpenMP parallelization */

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
   - Introduced restrict-qualified local pointers for ex, ey, hz, _fict_
     to help the compiler assume no aliasing and generate better
     vectorized code.
   - Hoisted scalar constants (0.5, 0.7) out of the loops.
   - Added OpenMP parallelization over the spatial loops, with the time
     loop kept sequential (per thread) to respect data dependencies.
   - Kept the memory access pattern (i outer, j inner) to preserve
     good cache locality for row-major storage.
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
  /* Create local restrict-qualified views of the arrays.
     This tells the compiler that these pointers do not alias each other,
     which is crucial for effective vectorization. */
  DATA_TYPE (* restrict ex_)[ny] = ex;
  DATA_TYPE (* restrict ey_)[ny] = ey;
  DATA_TYPE (* restrict hz_)[ny] = hz;
  DATA_TYPE * restrict fict_    = _fict_;

  /* Use PolyBench "prepared" loop bounds to allow possible runtime
     problem size changes while enabling the compiler to see constants. */
  const int tmax_ = _PB_TMAX;
  const int nx_   = _PB_NX;
  const int ny_   = _PB_NY;

  /* Hoist scalar constants out of the loops to avoid repeated
     conversions / loads. */
  const DATA_TYPE half         = SCALAR_VAL(0.5);
  const DATA_TYPE seven_tenths = SCALAR_VAL(0.7);

#pragma scop

  /* Parallel region over spatial work; the time loop remains logically
     sequential, but all spatial loops inside each timestep are
     distributed among threads. Each thread has its own private loop
     indices; the shared arrays hold the global state. */
#pragma omp parallel default(shared)
  {
    int i, j;

    /* t is private to each thread; all threads execute the same sequence
       of timesteps, synchronizing at the implicit barriers of each
       omp for. */
    for (int t = 0; t < tmax_; ++t)
    {
      /* 1. Update the first row of ey with the fictitious boundary
            condition for this timestep. */
#pragma omp for
      for (j = 0; j < ny_; j++)
        ey_[0][j] = fict_[t];

      /* 2. Update the remaining rows of ey using the hz field.
            No loop-carried dependencies across (i,j), so this is
            safe to parallelize and vectorize. */
#pragma omp for collapse(2)
      for (i = 1; i < nx_; i++)
        for (j = 0; j < ny_; j++)
          ey_[i][j] = ey_[i][j] - half * (hz_[i][j] - hz_[i-1][j]);

      /* 3. Update ex using hz. Again, independent across (i,j)
            for i in [0, nx_), j in [1, ny_). */
#pragma omp for collapse(2)
      for (i = 0; i < nx_; i++)
        for (j = 1; j < ny_; j++)
          ex_[i][j] = ex_[i][j] - half * (hz_[i][j] - hz_[i][j-1]);

      /* 4. Update hz using the newly computed ex and ey.
            Reading only from ex_ and ey_ and writing to hz_;
            each (i,j) update is independent. */
#pragma omp for collapse(2)
      for (i = 0; i < nx_ - 1; i++)
        for (j = 0; j < ny_ - 1; j++)
          hz_[i][j] = hz_[i][j] - seven_tenths *
                      ( (ex_[i][j+1] - ex_[i][j]) +
                        (ey_[i+1][j] - ey_[i][j]) );

      /* Implicit barrier at the end of the last omp for ensures that
         all updates for timestep t are complete before any thread
         proceeds to timestep t+1. */
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