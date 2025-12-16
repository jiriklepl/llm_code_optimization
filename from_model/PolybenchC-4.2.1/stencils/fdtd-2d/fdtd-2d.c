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

   Optimizations applied (while preserving original semantics):
   - Use local __restrict__ pointers for ex/ey/hz/_fict_ to aid the compiler
     with alias analysis and enable more aggressive vectorization.
   - Hoist scalar constants (0.5 and 0.7) out of inner loops.
   - Hoist _fict_[t] into a scalar per time step to avoid repeated loads.
   - Apply 2D spatial blocking (tiling) in i and j for the three main
     update sweeps to improve cache locality when working on large grids.
   - Within tiles, use row pointers so that the innermost j-loops have
     unit-stride memory access, which is friendly for SIMD vectorization.
   - The time loop t, and the order of algorithmic phases within each time
     step, remain unchanged: Ey boundary -> Ey update -> Ex update -> Hz update.
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
  int t, i, j;

  /* Local restrict-qualified views of the arrays.
     ex, ey, hz are distinct buffers in PolyBench, so the restrict
     assumptions are valid and help the optimizer. */
  DATA_TYPE (* __restrict ex_)[ny]   = ex;
  DATA_TYPE (* __restrict ey_)[ny]   = ey;
  DATA_TYPE (* __restrict hz_)[ny]   = hz;
  DATA_TYPE * __restrict fict_      = _fict_;

  /* Precompute scalar coefficients used in the stencil. */
  const DATA_TYPE cey = SCALAR_VAL(0.5);
  const DATA_TYPE chz = SCALAR_VAL(0.7);

  /* Tile sizes for spatial blocking.
     These values are chosen to keep 3 * TI * TJ cells (ex, ey, hz)
     comfortably within typical L1/L2 cache on modern x64 CPUs. */
  const int TI = 32;
  const int TJ = 256;

#pragma scop

  for (t = 0; t < _PB_TMAX; t++)
    {
      /* Hoist the fictitious boundary value for this time step. */
      const DATA_TYPE ft = fict_[t];

      /* 1) Ey boundary condition: ey[0][j] = _fict_[t] for all j.
         This loop is simple and already cache- and vector-friendly. */
      for (j = 0; j < _PB_NY; j++)
	ey_[0][j] = ft;

      /* 2) Ey update: for i in [1, NX), j in [0, NY)
             ey[i][j] -= 0.5 * (hz[i][j] - hz[i-1][j]);
         We apply 2D tiling in (i, j) to improve cache locality. */
      {
        const int nx_eff = _PB_NX;
        const int ny_eff = _PB_NY;
        int ii, jj;

        for (ii = 1; ii < nx_eff; ii += TI) {
          int i_end = ii + TI;
          if (i_end > nx_eff)
            i_end = nx_eff;

          for (jj = 0; jj < ny_eff; jj += TJ) {
            int j_end = jj + TJ;
            if (j_end > ny_eff)
              j_end = ny_eff;

            for (i = ii; i < i_end; ++i) {
              DATA_TYPE * __restrict ey_row      = ey_[i];
              DATA_TYPE * __restrict hz_row      = hz_[i];
              DATA_TYPE * __restrict hz_prev_row = hz_[i - 1];

              /* Inner j-loop has unit-stride accesses in all arrays.  */
              for (j = jj; j < j_end; ++j) {
                ey_row[j] -= cey * (hz_row[j] - hz_prev_row[j]);
              }
            }
          }
        }
      }

      /* 3) Ex update: for i in [0, NX), j in [1, NY)
             ex[i][j] -= 0.5 * (hz[i][j] - hz[i][j-1]);
         Again, apply 2D tiling. */
      {
        const int nx_eff = _PB_NX;
        const int ny_eff = _PB_NY;
        int ii, jj;

        for (ii = 0; ii < nx_eff; ii += TI) {
          int i_end = ii + TI;
          if (i_end > nx_eff)
            i_end = nx_eff;

          /* j starts from 1 to match original loop bounds. */
          for (jj = 1; jj < ny_eff; jj += TJ) {
            int j_start = jj;
            if (j_start < 1)
              j_start = 1;
            int j_end = jj + TJ;
            if (j_end > ny_eff)
              j_end = ny_eff;

            for (i = ii; i < i_end; ++i) {
              DATA_TYPE * __restrict ex_row = ex_[i];
              DATA_TYPE * __restrict hz_row = hz_[i];

              for (j = j_start; j < j_end; ++j) {
                ex_row[j] -= cey * (hz_row[j] - hz_row[j - 1]);
              }
            }
          }
        }
      }

      /* 4) Hz update: for i in [0, NX-1), j in [0, NY-1)
             hz[i][j] -= 0.7 *
                          ( (ex[i][j+1] - ex[i][j]) +
                            (ey[i+1][j]  - ey[i][j]) );
         Tiling again improves cache reuse of ex, ey, and hz. */
      {
        const int nx_eff = _PB_NX - 1; /* exclusive upper bound for i */
        const int ny_eff = _PB_NY - 1; /* exclusive upper bound for j */
        int ii, jj;

        for (ii = 0; ii < nx_eff; ii += TI) {
          int i_end = ii + TI;
          if (i_end > nx_eff)
            i_end = nx_eff;

          for (jj = 0; jj < ny_eff; jj += TJ) {
            int j_end = jj + TJ;
            if (j_end > ny_eff)
              j_end = ny_eff;

            for (i = ii; i < i_end; ++i) {
              DATA_TYPE * __restrict hz_row      = hz_[i];
              DATA_TYPE * __restrict ex_row      = ex_[i];
              DATA_TYPE * __restrict ey_row      = ey_[i];
              DATA_TYPE * __restrict ey_row_down = ey_[i + 1];

              for (j = jj; j < j_end; ++j) {
                const DATA_TYPE d_ex = ex_row[j + 1] - ex_row[j];
                const DATA_TYPE d_ey = ey_row_down[j] - ey_row[j];
                hz_row[j] -= chz * (d_ex + d_ey);
              }
            }
          }
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