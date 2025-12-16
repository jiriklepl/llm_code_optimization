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


/* Tunable parameter:
 *   FDTD_OMP_MIN_SIZE controls when OpenMP parallelism is enabled.
 *   For small problems the OpenMP overhead can dominate, so leaving
 *   the kernel single-threaded may be faster.
 *   Adjust this value for your target machine.
 */
#ifndef FDTD_OMP_MIN_SIZE
# define FDTD_OMP_MIN_SIZE 32768L
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return.

   Optimizations applied:
   - Use local restrict-qualified pointers (ex_, ey_, hz_) to tell the
     compiler that the main arrays do not alias, enabling aggressive
     vectorization and reordering.
   - Hoist scalar constants (0.5, 0.7) out of inner loops.
   - Keep innermost loops contiguous in memory (j-index), but add
     OpenMP parallelism across the outer dimension(s) when enabled.
   - Use #pragma omp simd on the inner loops to encourage SIMD
     vectorization even under conservative alias assumptions.

   The numerical scheme and iteration space are unchanged.
*/
static __attribute__((noinline, flatten))
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  /* tmax, nx, ny are also used implicitly through PolyBench macros
     (e.g., _PB_NX), but we cast them to void to silence potential
     unused-parameter warnings when those macros expand to constants. */
  (void)tmax;
  (void)nx;
  (void)ny;

  /* Local restrict-qualified views of the arrays.
   * These promise the compiler that ex_, ey_ and hz_ point to
   * non-overlapping storage, which is true for this benchmark.
   */
  DATA_TYPE (* __restrict ex_)[ny] = ex;
  DATA_TYPE (* __restrict ey_)[ny] = ey;
  DATA_TYPE (* __restrict hz_)[ny] = hz;
  DATA_TYPE * __restrict fict_    = _fict_;

  const DATA_TYPE half         = SCALAR_VAL(0.5);
  const DATA_TYPE seven_tenths = SCALAR_VAL(0.7);

  /* Shared scalar used to broadcast _fict_[t] to all OpenMP threads. */
  DATA_TYPE fict_t_shared = SCALAR_VAL(0.0);

#pragma scop

  /* One parallel region encompassing the whole time-stepping loop.
   * When OpenMP is disabled (no -fopenmp), all omp pragmas are ignored
   * and the code executes sequentially with the original semantics.
   */
#pragma omp parallel shared(fict_t_shared, ex_, ey_, hz_, fict_) \
                     if (((long)nx * (long)ny) >= FDTD_OMP_MIN_SIZE)
  {
    int t, i, j;

    for (t = 0; t < _PB_TMAX; t++)
      {
        /* Load the boundary value for this time step once and share it. */
#pragma omp single
        {
          fict_t_shared = fict_[t];
        }
        /* Implicit barrier at the end of 'single' ensures all threads
           see fict_t_shared before using it. */

        /* ey[0][j] = _fict_[t];  -- boundary condition on first row */
        DATA_TYPE * __restrict ey0 = ey_[0];

#pragma omp for schedule(static)
        for (j = 0; j < _PB_NY; j++)
          {
            ey0[j] = fict_t_shared;
          }
        /* Implicit barrier here: all ey[0][*] are updated before
           the next Ey update stage starts. */

        /* ey[i][j] -= 0.5 * (hz[i][j] - hz[i-1][j]);  for i >= 1 */
#pragma omp for schedule(static)
        for (i = 1; i < _PB_NX; i++)
          {
            DATA_TYPE       * __restrict ey_i   = ey_[i];
            const DATA_TYPE * __restrict hz_i   = hz_[i];
            const DATA_TYPE * __restrict hz_im1 = hz_[i-1];

#pragma omp simd
            for (j = 0; j < _PB_NY; j++)
              {
                ey_i[j] = ey_i[j] - half * (hz_i[j] - hz_im1[j]);
              }
          }
        /* Implicit barrier: all Ey values for this time step are ready
           before starting the Ex update. */

        /* ex[i][j] -= 0.5 * (hz[i][j] - hz[i][j-1]);  for j >= 1 */
#pragma omp for schedule(static)
        for (i = 0; i < _PB_NX; i++)
          {
            DATA_TYPE       * __restrict ex_i = ex_[i];
            const DATA_TYPE * __restrict hz_i = hz_[i];

#pragma omp simd
            for (j = 1; j < _PB_NY; j++)
              {
                ex_i[j] = ex_i[j] - half * (hz_i[j] - hz_i[j-1]);
              }
          }
        /* Implicit barrier: all Ex values for this time step are ready
           before updating Hz. */

        /* hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] +
                              ey[i+1][j] - ey[i][j]); */
#pragma omp for schedule(static)
        for (i = 0; i < _PB_NX - 1; i++)
          {
            DATA_TYPE       * __restrict hz_i   = hz_[i];
            const DATA_TYPE * __restrict ex_i   = ex_[i];
            const DATA_TYPE * __restrict ey_i   = ey_[i];
            const DATA_TYPE * __restrict ey_ip1 = ey_[i+1];

#pragma omp simd
            for (j = 0; j < _PB_NY - 1; j++)
              {
                hz_i[j] = hz_i[j] - seven_tenths *
                          ( (ex_i[j+1] - ex_i[j]) +
                            (ey_ip1[j]  - ey_i[j]) );
              }
          }
        /* Implicit barrier: Hz for this time step is complete before
           the loop proceeds to t+1. */

      } /* end of t loop */
  } /* end of parallel region */

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