/**
 * Exo FDTD-2D driver: mirrors PolyBench/C fdtd-2d.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"

/* Include the Exo-generated kernel header. */
#include "generated/fdtd-2d/fdtd-2d.h"


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


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM
from exo.API_scheduling import *

# Tile sizes chosen for good cache locality on a typical x64 CPU.
# They are compile-time constants in the Exo metaprogram.
TI = 32  # tile size in the i (row) dimension
TJ = 64  # tile size in the j (column) dimension


@proc
def kernel_fdtd_2d(
    tmax: size,
    nx: size,
    ny: size,
    ex: DATA_TYPE[nx, ny] @ DRAM,
    ey: DATA_TYPE[nx, ny] @ DRAM,
    hz: DATA_TYPE[nx, ny] @ DRAM,
    _fict_: DATA_TYPE[tmax] @ DRAM,
):
    # Basic sanity for the stencil; PolyBench uses much larger sizes.
    assert tmax >= 1
    assert nx  >= 2
    assert ny  >= 2

    for t in seq(0, tmax):
        # Phase 1: boundary condition on the first row of ey.
        # Hoist _fict_[t] into a scalar to avoid reloading per element.
        fict_t: DATA_TYPE
        fict_t = _fict_[t]

        for j in seq(0, ny):
            ey[0, j] = fict_t

        # Phase 2: Ey update for i in [1, nx), all j.
        #   ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
        for i in seq(1, nx):
            for j in seq(0, ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])

        # Phase 3: Ex update for all i, j in [1, ny).
        #   ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
        for i in seq(0, nx):
            for j in seq(1, ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])

        # Phase 4: Hz update on the interior region.
        #   hz[i, j] = hz[i, j] - 0.7 * (
        #                  ex[i, j + 1] - ex[i, j]
        #                + ey[i + 1, j] - ey[i, j])
        for i in seq(0, nx - 1):
            for j in seq(0, ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j]
                    + ey[i + 1, j] - ey[i, j]
                )


# --------------------------------------------------------------------------
# Scheduling: 2D spatial tiling and OpenMP-style parallelization.
#
# We keep the time loop sequential and only reorder/parallelize the
# spatial (i, j) loops inside each of the three update phases.
# --------------------------------------------------------------------------

# Tile and parallelize Phase 2 (Ey update).
# First "i" loop in the procedure is the Ey-update loop.
i_ey = kernel_fdtd_2d.find_loop("i")
# Its body contains exactly one inner "j" loop.
j_ey = i_ey.body().find_loop("j")

# 2D tiling of (i, j) for Ey update.
kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, i_ey, TI, ("io_ey", "ii_ey"), tail="guard")
kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, j_ey, TJ, ("jo_ey", "ji_ey"), tail="guard")

# Parallelize over tiles in the i-direction for Ey.
io_ey = kernel_fdtd_2d.find_loop("io_ey")
kernel_fdtd_2d = parallelize_loop(kernel_fdtd_2d, io_ey)


# Tile and parallelize Phase 3 (Ex update).
# After tiling Ey, the next "i" loop is the Ex-update loop.
i_ex = kernel_fdtd_2d.find_loop("i")
j_ex = i_ex.body().find_loop("j")

kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, i_ex, TI, ("io_ex", "ii_ex"), tail="guard")
kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, j_ex, TJ, ("jo_ex", "ji_ex"), tail="guard")

io_ex = kernel_fdtd_2d.find_loop("io_ex")
kernel_fdtd_2d = parallelize_loop(kernel_fdtd_2d, io_ex)


# Tile and parallelize Phase 4 (Hz update).
# The remaining "i" loop is the Hz-update loop.
i_hz = kernel_fdtd_2d.find_loop("i")
j_hz = i_hz.body().find_loop("j")

kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, i_hz, TI, ("io_hz", "ii_hz"), tail="guard")
kernel_fdtd_2d = divide_loop(kernel_fdtd_2d, j_hz, TJ, ("jo_hz", "ji_hz"), tail="guard")

io_hz = kernel_fdtd_2d.find_loop("io_hz")
kernel_fdtd_2d = parallelize_loop(kernel_fdtd_2d, io_hz)

EXO END
*/


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

  /* Run Exo kernel. Flatten PolyBench views to 1D pointers. */
  kernel_fdtd_2d(/*ctxt=*/NULL, tmax, nx, ny,
                 (DATA_TYPE*)POLYBENCH_ARRAY(ex),
                 (DATA_TYPE*)POLYBENCH_ARRAY(ey),
                 (DATA_TYPE*)POLYBENCH_ARRAY(hz),
                 (DATA_TYPE*)POLYBENCH_ARRAY(_fict_));

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