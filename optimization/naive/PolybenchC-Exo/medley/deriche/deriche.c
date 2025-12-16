/**
 * Exo Deriche driver: mirrors PolyBench/C deriche.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"

/* Include the Exo-generated kernel header. */
#include "generated/deriche/deriche.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}


/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Extern definition for exp
class _Exp(Extern):
    def __init__(self):
        super().__init__("exp")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        arg_type = args[0].type
        if not arg_type.is_real_scalar():
            raise _EErr(
                f"expected argument to be a real scalar value, but got type {arg_type}"
            )
        return arg_type

    def compile(self, args, prim_type):
        return f"exp(({prim_type}){args[0]})"

    def globl(self, prim_type):
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.exp(args[0])


# Extern definition for pow
class _Pow(Extern):
    def __init__(self):
        super().__init__("pow")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")

        arg_type0 = args[0].type
        arg_type1 = args[1].type
        if not arg_type0.is_real_scalar() or not arg_type1.is_real_scalar():
            raise _EErr(
                f"expected arguments to be real scalar values, but got types {arg_type0}, {arg_type1}"
            )
        # Use the type of the first argument as the result type.
        return arg_type0

    def compile(self, args, prim_type):
        return f"pow(({prim_type}){args[0]}, ({prim_type}){args[1]})"

    def globl(self, prim_type):
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.pow(args[0], args[1])


exp = _Exp()
pow = _Pow()


@proc
def deriche_base(
    w: size,
    h: size,
    alpha: DATA_TYPE,
    imgIn: DATA_TYPE[w, h] @ DRAM,
    imgOut: DATA_TYPE[w, h] @ DRAM,
    y1: DATA_TYPE[w, h] @ DRAM,
    y2: DATA_TYPE[w, h] @ DRAM,
):
    # Temporary scalars for the recursive filters
    xm1: DATA_TYPE
    ym1: DATA_TYPE
    ym2: DATA_TYPE
    xp1: DATA_TYPE
    xp2: DATA_TYPE
    yp1: DATA_TYPE
    yp2: DATA_TYPE

    # Coefficient scalars
    ema: DATA_TYPE
    ema2: DATA_TYPE
    exp2a: DATA_TYPE

    k: DATA_TYPE
    a1: DATA_TYPE
    a2: DATA_TYPE
    a3: DATA_TYPE
    a4: DATA_TYPE
    a5: DATA_TYPE
    a6: DATA_TYPE
    a7: DATA_TYPE
    a8: DATA_TYPE
    b1: DATA_TYPE
    b2: DATA_TYPE
    c1: DATA_TYPE
    c2: DATA_TYPE

    # Per-column state for horizontal passes (kept in contiguous 1D buffers)
    tm1_line: DATA_TYPE[h]
    ym1_line: DATA_TYPE[h]
    ym2_line: DATA_TYPE[h]
    tp1_line: DATA_TYPE[h]
    tp2_line: DATA_TYPE[h]
    yp1_line: DATA_TYPE[h]
    yp2_line: DATA_TYPE[h]

    # ------------------------------------------------------------
    # Coefficient computation (precompute exponentials once)
    # ------------------------------------------------------------
    ema = exp(-alpha)          # exp(-alpha)
    ema2 = ema * ema           # exp(-2*alpha)
    exp2a = exp(2.0 * alpha)   # exp(2*alpha)

    k = (
        (1.0 - ema) * (1.0 - ema)
        / (1.0 + 2.0 * alpha * ema - exp2a)
    )
    a1 = k
    a5 = k
    a2 = k * ema * (alpha - 1.0)
    a6 = a2
    a3 = k * ema * (alpha + 1.0)
    a7 = a3
    a4 = -k * ema2
    a8 = a4
    b1 = pow(2.0, -alpha)
    b2 = -ema2
    c1 = 1.0
    c2 = 1.0

    # ------------------------------------------------------------
    # Vertical filtering (forward + backward) and combine
    # Process one row at a time to keep imgIn/imgOut/y1/y2 row data hot.
    # ------------------------------------------------------------
    for i in seq(0, w):
        # Vertical forward pass along j
        xm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        for j in seq(0, h):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

        # Vertical backward pass along j (reverse direction)
        xp1 = 0.0
        xp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        for j in seq(0, h):
            y2[i, h - 1 - j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i, h - 1 - j]
            yp2 = yp1
            yp1 = y2[i, h - 1 - j]

        # Combine vertical passes into intermediate imgOut
        for j in seq(0, h):
            imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])

    # ------------------------------------------------------------
    # Horizontal forward pass
    # Reorganized to be row-major (i outer, j inner).
    # We keep per-column state in tm1_line/ym1_line/ym2_line so that
    # each column j maintains its own recurrence while we sweep rows.
    # This yields unit-stride accesses in j for imgOut, y1, and state.
    # ------------------------------------------------------------
    for j in seq(0, h):
        tm1_line[j] = 0.0
        ym1_line[j] = 0.0
        ym2_line[j] = 0.0

    for i in seq(0, w):
        for j in seq(0, h):
            y1[i, j] = a5 * imgOut[i, j] + a6 * tm1_line[j] + b1 * ym1_line[j] + b2 * ym2_line[j]
            tm1_line[j] = imgOut[i, j]
            ym2_line[j] = ym1_line[j]
            ym1_line[j] = y1[i, j]

    # ------------------------------------------------------------
    # Horizontal backward pass
    # Also row-major: we iterate rows from bottom to top using w-1-ii.
    # Per-column state (tp1_line/tp2_line/yp1_line/yp2_line) carries the
    # backward recurrence independently for each column.
    # ------------------------------------------------------------
    for j in seq(0, h):
        tp1_line[j] = 0.0
        tp2_line[j] = 0.0
        yp1_line[j] = 0.0
        yp2_line[j] = 0.0

    for ii in seq(0, w):
        for j in seq(0, h):
            y2[w - 1 - ii, j] = a7 * tp1_line[j] + a8 * tp2_line[j] + b1 * yp1_line[j] + b2 * yp2_line[j]
            tp2_line[j] = tp1_line[j]
            tp1_line[j] = imgOut[w - 1 - ii, j]
            yp2_line[j] = yp1_line[j]
            yp1_line[j] = y2[w - 1 - ii, j]

    # ------------------------------------------------------------
    # Final combine of horizontal passes
    # ------------------------------------------------------------
    for i in seq(0, w):
        for j in seq(0, h):
            imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])


# Apply a simple scheduling pass and export the final kernel
deriche_opt = simplify(deriche_base)
kernel_deriche = rename(deriche_opt, "kernel_deriche")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);

  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten 2D views to 1D pointers. */
  kernel_deriche(/*ctxt=*/NULL,
                 w, h,
                 (DATA_TYPE*)&alpha,
                 (DATA_TYPE*)POLYBENCH_ARRAY(imgIn),
                 (DATA_TYPE*)POLYBENCH_ARRAY(imgOut),
                 (DATA_TYPE*)POLYBENCH_ARRAY(y1),
                 (DATA_TYPE*)POLYBENCH_ARRAY(y2));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  return 0;
}