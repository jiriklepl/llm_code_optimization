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
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


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
        # For simplicity, use the type of the first argument as the result type.
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
def kernel_deriche(
    w: size,
    h: size,
    alpha: DATA_TYPE,
    imgIn: DATA_TYPE[w, h] @ DRAM,
    imgOut: DATA_TYPE[w, h] @ DRAM,
    y1: DATA_TYPE[w, h] @ DRAM,
    y2: DATA_TYPE[w, h] @ DRAM,
):
    xm1: DATA_TYPE
    tm1: DATA_TYPE
    ym1: DATA_TYPE
    ym2: DATA_TYPE
    xp1: DATA_TYPE
    xp2: DATA_TYPE
    tp1: DATA_TYPE
    tp2: DATA_TYPE
    yp1: DATA_TYPE
    yp2: DATA_TYPE

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

    # Coefficient computation
    k = (
        (1.0 - exp(-alpha)) * (1.0 - exp(-alpha))
        / (1.0 + 2.0 * alpha * exp(-alpha) - exp(2.0 * alpha))
    )
    a1 = k
    a5 = k
    a2 = k * exp(-alpha) * (alpha - 1.0)
    a6 = a2
    a3 = k * exp(-alpha) * (alpha + 1.0)
    a7 = a3
    a4 = -k * exp(-2.0 * alpha)
    a8 = a4
    b1 = pow(2.0, -alpha)
    b2 = -exp(-2.0 * alpha)
    c1 = 1.0
    c2 = 1.0

    # Vertical forward pass
    for i in seq(0, w):
        ym1 = 0.0
        ym2 = 0.0
        xm1 = 0.0
        for j in seq(0, h):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    # Vertical backward pass
    for i in seq(0, w):
        yp1 = 0.0
        yp2 = 0.0
        xp1 = 0.0
        xp2 = 0.0
        for j in seq(0, h):
            y2[i, h - 1 - j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i, h - 1 - j]
            yp2 = yp1
            yp1 = y2[i, h - 1 - j]

    # Combine vertical passes
    for i in seq(0, w):
        for j in seq(0, h):
            imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])

    # Horizontal forward pass
    for j in seq(0, h):
        tm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        for i in seq(0, w):
            y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = imgOut[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    # Horizontal backward pass
    for j in seq(0, h):
        tp1 = 0.0
        tp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        for i in seq(0, w):
            y2[w - 1 - i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = imgOut[w - 1 - i, j]
            yp2 = yp1
            yp1 = y2[w - 1 - i, j]

    # Final combine
    for i in seq(0, w):
        for j in seq(0, h):
            imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])
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