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
from exo.API_scheduling import *  # scheduling primitives (not used directly here but available)
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr

# ---------------------------------------------------------------------------
# Extern definitions for exp and pow
# ---------------------------------------------------------------------------
# These externs allow us to use math.h's exp() and pow() inside Exo object
# code. They are used only on RHS expressions, as required by Exo.
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

# Tile sizes for 2D pointwise loops. These are compile-time constants
# chosen to improve cache locality and expose long, contiguous inner
# loops in j that the C compiler can auto-vectorize.
TI = 64
TJ = 64


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
    # ------------------------------------------------------------------
    # Additional 1D temporaries for the horizontal passes
    # ------------------------------------------------------------------
    # We keep horizontal recursion state per column j in these small O(h)
    # buffers. This lets us run the horizontal passes in i-major, j-minor
    # order, so the inner loop walks contiguous memory (j) instead of
    # large-stride accesses in i. The extra memory is negligible compared
    # to the 2D images (well under 50% overhead).
    tm1_h: DATA_TYPE[h] @ DRAM
    ym1_h: DATA_TYPE[h] @ DRAM
    ym2_h: DATA_TYPE[h] @ DRAM
    tp1_h: DATA_TYPE[h] @ DRAM
    tp2_h: DATA_TYPE[h] @ DRAM
    yp1_h: DATA_TYPE[h] @ DRAM
    yp2_h: DATA_TYPE[h] @ DRAM

    # Scalar temporaries for the vertical recurrences.
    xm1: DATA_TYPE
    ym1: DATA_TYPE
    ym2: DATA_TYPE
    xp1: DATA_TYPE
    xp2: DATA_TYPE
    yp1: DATA_TYPE
    yp2: DATA_TYPE

    # Filter coefficients.
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

    # Hoisted exponentials to avoid redundant transcendental calls.
    em: DATA_TYPE   # exp(-alpha)
    em2: DATA_TYPE  # exp(-2*alpha)
    ep2: DATA_TYPE  # exp( 2*alpha)

    # ------------------------------------------------------------------
    # Coefficient computation (once per call).
    # We explicitly reuse exp(-alpha), exp(-2*alpha), and exp(2*alpha)
    # instead of recomputing them at every use site, reducing the
    # cost of transcendental operations while preserving the exact
    # formulas of the original kernel.
    # ------------------------------------------------------------------
    em = exp(-alpha)
    em2 = exp(-2.0 * alpha)
    ep2 = exp(2.0 * alpha)

    k = ((1.0 - em) * (1.0 - em)) / (1.0 + 2.0 * alpha * em - ep2)

    a1 = k
    a5 = k

    a2 = k * em * (alpha - 1.0)
    a6 = a2

    a3 = k * em * (alpha + 1.0)
    a7 = a3

    a4 = -k * em2
    a8 = a4

    b1 = pow(2.0, -alpha)
    b2 = -em2

    c1 = 1.0
    c2 = 1.0

    # ------------------------------------------------------------------
    # Vertical forward pass: causal IIR filter along j (top -> bottom)
    # for each fixed column i. Data layout is img[i, j], with j
    # contiguous, so this loop already has good locality.
    # ------------------------------------------------------------------
    for i in seq(0, w):
        xm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        for j in seq(0, h):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    # ------------------------------------------------------------------
    # Vertical backward pass: anti-causal IIR filter along j (bottom -> top)
    # for each fixed column i. We keep the structure of the original
    # implementation for clarity; it also enjoys unit-stride access in j.
    # ------------------------------------------------------------------
    for i in seq(0, w):
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

    # ------------------------------------------------------------------
    # Combine vertical passes into an intermediate image in imgOut.
    #
    # This is a pure 2D map: each (i, j) depends only on y1[i, j] and
    # y2[i, j]. We tile the iteration space so that we work on cache-
    # friendly tiles and the inner loop over j remains contiguous.
    # ------------------------------------------------------------------
    for II in seq(0, (w + TI - 1) / TI):
        for JJ in seq(0, (h + TJ - 1) / TJ):
            for i in seq(0, TI):
                if II * TI + i < w:
                    for j in seq(0, TJ):
                        if JJ * TJ + j < h:
                            imgOut[II * TI + i, JJ * TJ + j] = (
                                c1
                                * (
                                    y1[II * TI + i, JJ * TJ + j]
                                    + y2[II * TI + i, JJ * TJ + j]
                                )
                            )

    # ------------------------------------------------------------------
    # Horizontal forward pass, optimized for locality.
    #
    # Original code:
    #   for j in seq(0, h):
    #       tm1 = ym1 = ym2 = 0
    #       for i in seq(0, w):
    #           y1[i, j] = ...
    #
    # This walks imgOut/y1 with a large stride in memory (since j is the
    # innermost/contiguous dimension).
    #
    # New code:
    #   - We keep per-column state (for each j) in 1D arrays tm1_h,
    #     ym1_h, ym2_h.
    #   - We iterate i-major / j-minor:
    #         for i in 0..w-1:
    #             for j in 0..h-1:
    #                 ...
    #   - For each fixed j, the sequence of i-values and the recurrence
    #     relation are *exactly* the same as in the original kernel, so
    #     the numerical result is unchanged.
    #   - Different columns j are independent and appear in the inner
    #     contiguous loop, which improves cache behavior and allows the
    #     C compiler to vectorize across j.
    # ------------------------------------------------------------------
    for j in seq(0, h):
        tm1_h[j] = 0.0
        ym1_h[j] = 0.0
        ym2_h[j] = 0.0

    for i in seq(0, w):
        for j in seq(0, h):
            y1[i, j] = (
                a5 * imgOut[i, j]
                + a6 * tm1_h[j]
                + b1 * ym1_h[j]
                + b2 * ym2_h[j]
            )
            tm1_h[j] = imgOut[i, j]
            ym2_h[j] = ym1_h[j]
            ym1_h[j] = y1[i, j]

    # ------------------------------------------------------------------
    # Horizontal backward pass, similarly optimized.
    #
    # Original code:
    #   for j in seq(0, h):
    #       tp1 = tp2 = yp1 = yp2 = 0
    #       for i in seq(0, w):
    #           ii = w - 1 - i
    #           y2[ii, j] = ...
    #
    # We now keep per-column state in 1D arrays tp1_h, tp2_h, yp1_h, yp2_h
    # and sweep i from right to left using i_rev, while still having j as
    # the inner contiguous loop. For each j, the sequence of spatial
    # indices visited is still (w-1, w-2, ..., 0), preserving the original
    # anti-causal recurrence exactly.
    # ------------------------------------------------------------------
    for j in seq(0, h):
        tp1_h[j] = 0.0
        tp2_h[j] = 0.0
        yp1_h[j] = 0.0
        yp2_h[j] = 0.0

    for i_rev in seq(0, w):
        for j in seq(0, h):
            y2[w - 1 - i_rev, j] = (
                a7 * tp1_h[j]
                + a8 * tp2_h[j]
                + b1 * yp1_h[j]
                + b2 * yp2_h[j]
            )
            tp2_h[j] = tp1_h[j]
            tp1_h[j] = imgOut[w - 1 - i_rev, j]
            yp2_h[j] = yp1_h[j]
            yp1_h[j] = y2[w - 1 - i_rev, j]

    # ------------------------------------------------------------------
    # Final combine of horizontal passes into imgOut.
    #
    # Like the vertical combine, this is a 2D pointwise operation
    # with no inter-iteration dependencies, so we tile it to improve
    # cache behavior and to keep the innermost j loop contiguous.
    # ------------------------------------------------------------------
    for II in seq(0, (w + TI - 1) / TI):
        for JJ in seq(0, (h + TJ - 1) / TJ):
            for i in seq(0, TI):
                if II * TI + i < w:
                    for j in seq(0, TJ):
                        if JJ * TJ + j < h:
                            imgOut[II * TI + i, JJ * TJ + j] = (
                                c2
                                * (
                                    y1[II * TI + i, JJ * TJ + j]
                                    + y2[II * TI + i, JJ * TJ + j]
                                )
                            )
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