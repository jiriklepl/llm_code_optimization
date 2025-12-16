/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"


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



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest
 *
 * Optimizations applied:
 *  - Precompute exponentials and filter coefficients once.
 *  - Add restrict-qualified local aliases to help the compiler
 *    with alias analysis and vectorization.
 *  - Fuse horizontal passes (forward, backward, combine) per row
 *    to improve temporal locality and reduce memory traffic.
 *  - Fuse vertical passes (forward, backward, combine) per column
 *    to reduce memory traffic.
 *  - Parallelize independent outer loops with OpenMP pragmas.
 *    (If compiled without OpenMP support, these pragmas are ignored
 *     and the code runs sequentially with identical semantics.)
 */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h))
{
  /* Create restrict-qualified aliases for better optimization.
     The PolyBench allocation ensures these arrays do not alias. */
  DATA_TYPE (* restrict imgIn_r )[h] = imgIn;
  DATA_TYPE (* restrict imgOut_r)[h] = imgOut;
  DATA_TYPE (* restrict y1_r   )[h] = y1;
  DATA_TYPE (* restrict y2_r   )[h] = y2;

  const DATA_TYPE one = SCALAR_VAL(1.0);
  const DATA_TYPE two = SCALAR_VAL(2.0);

  int i, j;

#pragma scop
  /* Precompute constants. We explicitly factor common subexpressions
     to reduce the number of calls to expensive transcendental
     functions (exp, pow) while preserving numerical semantics. */
  const DATA_TYPE ema  = EXP_FUN(-alpha);              /* e^{-alpha}   */
  const DATA_TYPE ema2 = EXP_FUN(-two * alpha);        /* e^{-2*alpha} */
  const DATA_TYPE e2a  = EXP_FUN( two * alpha);        /* e^{ 2*alpha} */

  const DATA_TYPE k =
    (one - ema) * (one - ema) /
    (one + two * alpha * ema - e2a);

  const DATA_TYPE a1 = k;
  const DATA_TYPE a2 = k * ema * (alpha - one);
  const DATA_TYPE a3 = k * ema * (alpha + one);
  const DATA_TYPE a4 = -k * ema2;

  /* By design of the Deriche filter, these are duplicates. */
  const DATA_TYPE a5 = a1;
  const DATA_TYPE a6 = a2;
  const DATA_TYPE a7 = a3;
  const DATA_TYPE a8 = a4;

  const DATA_TYPE b1 = POW_FUN(two, -alpha);
  const DATA_TYPE b2 = -ema2;

  const DATA_TYPE c1 = one;
  const DATA_TYPE c2 = one;

  /* -------------------- Horizontal pass --------------------
   * For each row i:
   *   1) forward (causal) recursion from left to right -> y1
   *   2) backward (anti-causal) recursion from right to left -> y2
   *   3) combine y1 and y2 into intermediate imgOut
   *
   * Each row is independent, so we parallelize over i.
   */
#pragma omp parallel for private(j) schedule(static) if (_PB_W * _PB_H > 1024*1024)
  for (i = 0; i < _PB_W; ++i)
  {
    /* Row pointers improve locality and keep base addresses in
       registers for the entire inner loops. */
    DATA_TYPE * restrict imgIn_row  = imgIn_r[i];
    DATA_TYPE * restrict y1_row     = y1_r[i];
    DATA_TYPE * restrict y2_row     = y2_r[i];
    DATA_TYPE * restrict imgOut_row = imgOut_r[i];

    /* Forward recursion (left-to-right). */
    DATA_TYPE xm1 = SCALAR_VAL(0.0);
    DATA_TYPE ym1 = SCALAR_VAL(0.0);
    DATA_TYPE ym2 = SCALAR_VAL(0.0);

    for (j = 0; j < _PB_H; ++j)
    {
      DATA_TYPE x = imgIn_row[j];
      DATA_TYPE y = a1 * x + a2 * xm1 + b1 * ym1 + b2 * ym2;

      y1_row[j] = y;

      xm1 = x;
      ym2 = ym1;
      ym1 = y;
    }

    /* Backward recursion (right-to-left) + combination.
       This fuses the original second (backward) and third
       (combine) horizontal loops, improving data locality
       and avoiding an extra sweep over imgOut. */
    DATA_TYPE xp1 = SCALAR_VAL(0.0);
    DATA_TYPE xp2 = SCALAR_VAL(0.0);
    DATA_TYPE yp1 = SCALAR_VAL(0.0);
    DATA_TYPE yp2 = SCALAR_VAL(0.0);

    for (j = _PB_H - 1; j >= 0; --j)
    {
      DATA_TYPE y = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;

      y2_row[j] = y;

      xp2 = xp1;
      xp1 = imgIn_row[j];
      yp2 = yp1;
      yp1 = y;

      /* Same formula as original separate combination loop. */
      imgOut_row[j] = c1 * (y1_row[j] + y);
    }
  }

  /* --------------------- Vertical pass ---------------------
   * Now work on columns using imgOut (result of horizontal pass)
   * as input:
   *
   *   1) forward recursion top-to-bottom -> y1
   *   2) backward recursion bottom-to-top -> y2
   *   3) final combination into imgOut (in-place)
   *
   * Each column j is independent, so we parallelize over j.
   * Note: memory access along i is inherently strided because
   *       of the row-major layout; fusing the passes reduces the
   *       total number of strided sweeps.
   */
#pragma omp parallel for private(i) schedule(static) if (_PB_W * _PB_H > 1024*1024)
  for (j = 0; j < _PB_H; ++j)
  {
    /* Forward recursion (top-to-bottom) for column j. */
    DATA_TYPE tm1 = SCALAR_VAL(0.0);
    DATA_TYPE vym1 = SCALAR_VAL(0.0);
    DATA_TYPE vym2 = SCALAR_VAL(0.0);

    for (i = 0; i < _PB_W; ++i)
    {
      DATA_TYPE * restrict imgOut_row = imgOut_r[i];
      DATA_TYPE * restrict y1_row     = y1_r[i];

      DATA_TYPE x = imgOut_row[j];
      DATA_TYPE y = a5 * x + a6 * tm1 + b1 * vym1 + b2 * vym2;

      y1_row[j] = y;

      tm1  = x;
      vym2 = vym1;
      vym1 = y;
    }

    /* Backward recursion (bottom-to-top) + final combination
       for column j. This fuses the original fifth and sixth
       loops, reducing memory traffic and improving locality
       for y1/y2/imgOut. */
    DATA_TYPE tp1 = SCALAR_VAL(0.0);
    DATA_TYPE tp2 = SCALAR_VAL(0.0);
    DATA_TYPE vyp1 = SCALAR_VAL(0.0);
    DATA_TYPE vyp2 = SCALAR_VAL(0.0);

    for (i = _PB_W - 1; i >= 0; --i)
    {
      DATA_TYPE * restrict imgOut_row = imgOut_r[i];
      DATA_TYPE * restrict y1_row     = y1_r[i];
      DATA_TYPE * restrict y2_row     = y2_r[i];

      DATA_TYPE y = a7 * tp1 + a8 * tp2 + b1 * vyp1 + b2 * vyp2;

      y2_row[j] = y;

      tp2  = tp1;
      tp1  = imgOut_row[j];
      vyp2 = vyp1;
      vyp1 = y;

      imgOut_row[j] = c2 * (y1_row[j] + y);
    }
  }

#pragma endscop
}


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

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

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