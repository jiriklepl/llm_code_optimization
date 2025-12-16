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
 * Optimized version:
 *  - Hoists common exponentials out of coefficient expressions.
 *  - Uses restrict-qualified row pointers for better alias analysis.
 *  - Fuses the three horizontal passes per row to improve cache reuse.
 *  - Restructures the vertical passes to iterate in row-major order,
 *    using small per-column state arrays. This removes the original
 *    strided (column-major) access pattern and enables contiguous
 *    streaming over imgOut/y1/y2 while preserving the 1D recurrences
 *    along columns exactly.
 */
static
void kernel_deriche[[gnu::flatten, gnu::noinline]](int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h))
{
  /* Create local restrict-qualified aliases to help the optimizer.
     POLYBENCH_2D(...) expands to a VLA parameter of type DATA_TYPE [w][h]. */
  DATA_TYPE (*restrict imgIn_)[h]  = imgIn;
  DATA_TYPE (*restrict imgOut_)[h] = imgOut;
  DATA_TYPE (*restrict y1_)[h]     = y1;
  DATA_TYPE (*restrict y2_)[h]     = y2;

  int i, j;

  DATA_TYPE k;
  DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
  DATA_TYPE b1, b2, c1, c2;

#pragma scop
  /* Precompute exponentials and powers once. This reduces the number
     of transcendental calls compared to repeatedly invoking EXP_FUN. */
  const DATA_TYPE em1  = EXP_FUN(-alpha);                            /* e^{-alpha}   */
  const DATA_TYPE em2a = EXP_FUN(SCALAR_VAL(-2.0) * alpha);          /* e^{-2alpha}  */
  const DATA_TYPE e2a  = EXP_FUN(SCALAR_VAL( 2.0) * alpha);          /* e^{ 2alpha}  */

  k  = (SCALAR_VAL(1.0) - em1) * (SCALAR_VAL(1.0) - em1) /
       (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * em1 - e2a);

  a1 = k;
  a5 = k;

  a2 = k * em1 * (alpha - SCALAR_VAL(1.0));
  a6 = a2;

  a3 = k * em1 * (alpha + SCALAR_VAL(1.0));
  a7 = a3;

  a4 = -k * em2a;
  a8 = a4;

  b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
  b2 = -em2a;

  c1 = SCALAR_VAL(1.0);
  c2 = SCALAR_VAL(1.0);

  /* ----------------------- Horizontal passes ----------------------- */
  /* For each row i:
   *   - run the forward (causal) horizontal recursion,
   *   - run the backward (anti-causal) horizontal recursion,
   *   - combine y1 and y2 into an intermediate imgOut row.
   * Rows are independent, and each inner loop over j walks
   * contiguous memory. */
  for (i = 0; i < _PB_W; ++i)
  {
    DATA_TYPE *restrict imgIn_row  = imgIn_[i];
    DATA_TYPE *restrict imgOut_row = imgOut_[i];
    DATA_TYPE *restrict y1_row     = y1_[i];
    DATA_TYPE *restrict y2_row     = y2_[i];

    /* Forward recursion (left to right) on row i. */
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

    /* Backward recursion (right to left) on row i. */
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
    }

    /* Horizontal combination for row i. */
    for (j = 0; j < _PB_H; ++j)
      imgOut_row[j] = c1 * (y1_row[j] + y2_row[j]);
  }

  /* ---------------- Vertical forward pass (top to bottom) ----------------
   *
   * Original code processes each column separately with an inner loop over i,
   * which results in strided (column-major) accesses on a row-major array.
   *
   * Here we keep the exact same per-column recurrences, but:
   *   - we iterate in row-major order (i outer, j inner),
   *   - for each column j we store its scalar state (tm1, ym1, ym2)
   *     in small 1D arrays v_tm1/v_ym1/v_ym2 of length h.
   *
   * This preserves the mathematical recurrences while turning memory
   * access to imgOut and y1 into contiguous streams. */
  {
    DATA_TYPE v_tm1[_PB_H];
    DATA_TYPE v_ym1[_PB_H];
    DATA_TYPE v_ym2[_PB_H];

    /* Zero-initialize per-column state (equivalent to tm1=ym1=ym2=0
       at the top of each original column loop). */
    for (j = 0; j < _PB_H; ++j)
    {
      v_tm1[j] = SCALAR_VAL(0.0);
      v_ym1[j] = SCALAR_VAL(0.0);
      v_ym2[j] = SCALAR_VAL(0.0);
    }

    /* Causal vertical recursion, column-wise dependence carried
       through the v_* arrays, but iterated in row-major order. */
    for (i = 0; i < _PB_W; ++i)
    {
      DATA_TYPE *restrict imgOut_row = imgOut_[i];
      DATA_TYPE *restrict y1_row     = y1_[i];

      for (j = 0; j < _PB_H; ++j)
      {
        const DATA_TYPE tm1  = v_tm1[j];
        const DATA_TYPE ym1v = v_ym1[j];
        const DATA_TYPE ym2v = v_ym2[j];
        DATA_TYPE x          = imgOut_row[j];

        DATA_TYPE y = a5 * x + a6 * tm1 + b1 * ym1v + b2 * ym2v;
        y1_row[j] = y;

        v_tm1[j] = x;
        v_ym2[j] = ym1v;
        v_ym1[j] = y;
      }
    }
  }

  /* --------------- Vertical backward pass + final combine ---------------
   *
   * We similarly restructure the anti-causal vertical recursion to run
   * in row-major order. For each column j we keep its backward state
   * (tp1, tp2, yp1, yp2) in 1D arrays v_tp1/v_tp2/v_yp1/v_yp2.
   *
   * We process rows in logical order i_b = _PB_W-1 .. 0 (bottom to top)
   * while still iterating the outer loop as i = 0 .. _PB_W-1. For each
   * (i_b, j) we:
   *   - compute y2[i_b][j] using the current backward state,
   *   - update the state from the original horizontally-filtered imgOut,
   *   - immediately form the final output imgOut[i_b][j].
   *
   * Note: We always read imgOut[i_b][j] into a temporary before
   * overwriting it, so the recurrence uses the same inputs as in the
   * original code (where vertical backward ran before the final combine). */
  {
    DATA_TYPE v_tp1[_PB_H];
    DATA_TYPE v_tp2[_PB_H];
    DATA_TYPE v_yp1[_PB_H];
    DATA_TYPE v_yp2[_PB_H];

    /* Zero-initialize per-column backward state. */
    for (j = 0; j < _PB_H; ++j)
    {
      v_tp1[j] = SCALAR_VAL(0.0);
      v_tp2[j] = SCALAR_VAL(0.0);
      v_yp1[j] = SCALAR_VAL(0.0);
      v_yp2[j] = SCALAR_VAL(0.0);
    }

    for (i = 0; i < _PB_W; ++i)
    {
      const int ib = _PB_W - 1 - i; /* logical index: W-1, W-2, ..., 0 */
      DATA_TYPE *restrict imgOut_row = imgOut_[ib];
      DATA_TYPE *restrict y1_row     = y1_[ib];
      DATA_TYPE *restrict y2_row     = y2_[ib];

      for (j = 0; j < _PB_H; ++j)
      {
        const DATA_TYPE tp1  = v_tp1[j];
        const DATA_TYPE tp2  = v_tp2[j];
        const DATA_TYPE yp1v = v_yp1[j];
        const DATA_TYPE yp2v = v_yp2[j];

        DATA_TYPE y = a7 * tp1 + a8 * tp2 + b1 * yp1v + b2 * yp2v;
        y2_row[j] = y;

        /* Update backward column state using the original horizontally
           filtered imgOut value at (ib, j). */
        DATA_TYPE x = imgOut_row[j];

        v_tp2[j] = tp1;
        v_tp1[j] = x;
        v_yp2[j] = yp1v;
        v_yp1[j] = y;

        /* Final combination for (ib, j). */
        imgOut_row[j] = c2 * (y1_row[j] + y);
      }
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