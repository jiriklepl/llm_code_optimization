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
from exo.API_scheduling import *
from exo.libs.memories import DRAM


# Base (unscheduled) FDTD-2D kernel matching the PolyBench reference.
@proc
def kernel_fdtd_2d_base(
    tmax: size,
    nx: size,
    ny: size,
    ex: DATA_TYPE[nx, ny] @ DRAM,
    ey: DATA_TYPE[nx, ny] @ DRAM,
    hz: DATA_TYPE[nx, ny] @ DRAM,
    _fict_: DATA_TYPE[tmax] @ DRAM,
):
    # Sanity constraints implied by the stencil (interior needs neighbors).
    assert nx > 1
    assert ny > 1

    for t in seq(0, tmax):
        # Ey boundary condition at i = 0.
        for j in seq(0, ny):
            ey[0, j] = _fict_[t]

        # Ey update: derivative along i-direction (uses hz[i, j] - hz[i-1, j]).
        for i in seq(1, nx):
            for j in seq(0, ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])

        # Ex update: derivative along j-direction (uses hz[i, j] - hz[i, j-1]).
        for i in seq(0, nx):
            for j in seq(1, ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])

        # Hz update: curl of the E field, uses updated Ex and Ey.
        for i in seq(0, nx - 1):
            for j in seq(0, ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j]
                    + ey[i + 1, j] - ey[i, j]
                )


# ---------------------------------------------------------------------------
# Scheduling: expose parallelism and improve spatial locality / ILP
# ---------------------------------------------------------------------------

# Public entry point: scheduled version of the base kernel.
kernel_fdtd_2d = rename(kernel_fdtd_2d_base, "kernel_fdtd_2d")

# Tile size for the innermost j-dimension. This gives a fixed-size loop
# that is friendly to unrolling and vectorization while still handling
# arbitrary ny via a guarded tail.
J_TILE = 32

# Work on a local variable and commit back to kernel_fdtd_2d at the end.
p = kernel_fdtd_2d

# 1) Parallelize the three large 2-D update loops over the i-dimension.
#    Each (i, j) update is independent within a time step, so these loops
#    are safely parallelizable.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)

# i_loops[0]: Ey interior update
p = parallelize_loop(p, i_loops[0])

# i_loops[1]: Ex update
p = parallelize_loop(p, i_loops[1])

# i_loops[2]: Hz update
p = parallelize_loop(p, i_loops[2])

# 2) Ey update: tile and unroll the j loop.
#    Layout is row-major in j, so strip-mining j improves cache behavior
#    and gives the compiler a regular inner loop.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
ey_i = i_loops[0]
ey_j = ey_i.body().find("for j in _:_")

# Strip-mine j into (jo, ji) with a guarded tail to handle ny % J_TILE != 0.
p = divide_loop(p, ey_j, J_TILE, ("jo", "ji"), tail="guard")

# Unroll the fixed-size inner loop ji to expose ILP and aid auto-vectorization.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
ey_i = i_loops[0]
jo = ey_i.body().find("for jo in _:_")
ji = jo.body().find("for ji in _:_")
p = unroll_loop(p, ji)

# 3) Ex update: shift j to start at 0, then tile and unroll.
#    divide_loop expects a 0-based loop; shift_loop keeps the iteration
#    set {1, ..., ny-1} but rewrites the index expression so the loop runs
#    from 0 to ny-2.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
ex_i = i_loops[1]
ex_j = ex_i.body().find("for j in _:_")

# Shift lower bound from 1 to 0 while preserving semantics.
p = shift_loop(p, ex_j, 0)

# Re-acquire and tile the shifted j loop.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
ex_i = i_loops[1]
ex_j = ex_i.body().find("for j in _:_")
p = divide_loop(p, ex_j, J_TILE, ("jo", "ji"), tail="guard")

# Unroll the inner ji loop for Ex update.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
ex_i = i_loops[1]
jo = ex_i.body().find("for jo in _:_")
ji = jo.body().find("for ji in _:_")
p = unroll_loop(p, ji)

# 4) Hz update: tile and unroll the j loop.
t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
hz_i = i_loops[2]
hz_j = hz_i.body().find("for j in _:_")

p = divide_loop(p, hz_j, J_TILE, ("jo", "ji"), tail="guard")

t_loop = p.find_loop("t")
t_body = t_loop.body()
i_loops = t_body.find("for i in _:_", many=True)
hz_i = i_loops[2]
jo = hz_i.body().find("for jo in _:_")
ji = jo.body().find("for ji in _:_")
p = unroll_loop(p, ji)

# Commit the scheduled kernel.
kernel_fdtd_2d = p
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