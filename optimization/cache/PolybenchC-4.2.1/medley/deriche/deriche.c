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
#include <stddef.h>  /* for size_t */

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"

/* -------------------------------------------------------------------------
 * Tuning constants
 * -------------------------------------------------------------------------
 *
 * DERICHE_OMP_MIN_SIZE controls when OpenMP parallelisation is enabled for
 * the large, embarrassingly parallel loops.  It is expressed in number of
 * pixels (w*h).  It can be overridden at compile time, e.g.:
 *
 *   -DDERICHE_OMP_MIN_SIZE=65536
 */
#ifndef DERICHE_OMP_MIN_SIZE
#define DERICHE_OMP_MIN_SIZE (1 << 18) /* 262144 elements by default */
#endif


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
/* Original code provided by Gael Deest */
static
void kernel_deriche [[gnu::flatten, gnu::noinline]](int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {

  /* Create restrict-qualified local aliases of the 2D arrays.
     This preserves the original interface while giving the compiler
     stronger aliasing information for optimisation.  We never access
     the original parameters directly below, only these aliases. */
  DATA_TYPE (* restrict imgInR)[h]  = imgIn;
  DATA_TYPE (* restrict imgOutR)[h] = imgOut;
  DATA_TYPE (* restrict y1R)[h]     = y1;
  DATA_TYPE (* restrict y2R)[h]     = y2;

#pragma scop
  const int W_ = _PB_W;
  const int H_ = _PB_H;

  /* Pre-compute filter coefficients.
     All expressions are kept algebraically identical to the original
     code; only common subexpressions are factored out. */
  const DATA_TYPE one = SCALAR_VAL(1.0);
  const DATA_TYPE two = SCALAR_VAL(2.0);

  const DATA_TYPE exp_m_alpha  = EXP_FUN(-alpha);
  const DATA_TYPE exp_p2_alpha = EXP_FUN(two * alpha);                 /* exp( 2*alpha )  */
  const DATA_TYPE exp_m2_alpha = EXP_FUN(SCALAR_VAL(-2.0) * alpha);    /* exp(-2*alpha )  */

  const DATA_TYPE k =
      (one - exp_m_alpha) * (one - exp_m_alpha) /
      (one + two * alpha * exp_m_alpha - exp_p2_alpha);

  const DATA_TYPE a1 = k;
  const DATA_TYPE a5 = k;

  const DATA_TYPE a2 = k * exp_m_alpha * (alpha - one);
  const DATA_TYPE a6 = a2;

  const DATA_TYPE a3 = k * exp_m_alpha * (alpha + one);
  const DATA_TYPE a7 = a3;

  const DATA_TYPE a4 = -k * exp_m2_alpha;
  const DATA_TYPE a8 = a4;

  const DATA_TYPE b1 = POW_FUN(two, -alpha);
  const DATA_TYPE b2 = -exp_m2_alpha;
  const DATA_TYPE c1 = SCALAR_VAL(1.0);
  const DATA_TYPE c2 = SCALAR_VAL(1.0);

  const size_t total_size = (size_t)W_ * (size_t)H_;

  /* --------------------------------------------------------------------
   * 1. Horizontal recursive filter (forward direction)
   *    y1[i][j] depends only on imgIn[i][*] and previous y1[i][*]
   *    along the row, so rows are independent and can be parallelised.
   *    Access pattern is row-major contiguous.
   * ------------------------------------------------------------------*/
#pragma omp parallel for if (total_size >= DERICHE_OMP_MIN_SIZE) schedule(static)
  for (int i = 0; i < W_; i++) {
    const DATA_TYPE * restrict imgIn_row = imgInR[i];
    DATA_TYPE * restrict y1_row = y1R[i];

    DATA_TYPE xm1 = SCALAR_VAL(0.0);
    DATA_TYPE ym1 = SCALAR_VAL(0.0);
    DATA_TYPE ym2 = SCALAR_VAL(0.0);

    for (int j = 0; j < H_; j++) {
      const DATA_TYPE x = imgIn_row[j];
      const DATA_TYPE y = a1 * x + a2 * xm1 + b1 * ym1 + b2 * ym2;
      y1_row[j] = y;

      xm1 = x;
      ym2 = ym1;
      ym1 = y;
    }
  }

  /* --------------------------------------------------------------------
   * 2. Horizontal recursive filter (reverse direction)
   *    Same as above, but processed right-to-left within each row and
   *    stored in y2.  Rows are again independent.
   * ------------------------------------------------------------------*/
#pragma omp parallel for if (total_size >= DERICHE_OMP_MIN_SIZE) schedule(static)
  for (int i = 0; i < W_; i++) {
    const DATA_TYPE * restrict imgIn_row = imgInR[i];
    DATA_TYPE * restrict y2_row = y2R[i];

    DATA_TYPE xp1 = SCALAR_VAL(0.0);
    DATA_TYPE xp2 = SCALAR_VAL(0.0);
    DATA_TYPE yp1 = SCALAR_VAL(0.0);
    DATA_TYPE yp2 = SCALAR_VAL(0.0);

    for (int j = H_ - 1; j >= 0; j--) {
      const DATA_TYPE y = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      y2_row[j] = y;

      xp2 = xp1;
      xp1 = imgIn_row[j];
      yp2 = yp1;
      yp1 = y;
    }
  }

  /* --------------------------------------------------------------------
   * 3. Combine horizontal passes
   *    Purely element-wise, trivially parallel and vectorisable.
   * ------------------------------------------------------------------*/
#pragma omp parallel for if (total_size >= DERICHE_OMP_MIN_SIZE) schedule(static)
  for (int i = 0; i < W_; i++) {
    const DATA_TYPE * restrict y1_row = y1R[i];
    const DATA_TYPE * restrict y2_row = y2R[i];
    DATA_TYPE * restrict out_row = imgOutR[i];

    for (int j = 0; j < H_; j++)
      out_row[j] = c1 * (y1_row[j] + y2_row[j]);
  }

  /* --------------------------------------------------------------------
   * 4. Vertical recursive filter (forward direction)
   *
   * Original code:
   *   for (j)
   *     for (i)  // recurrence along i, stride-H_ memory accesses
   *
   * This has poor spatial locality, because successive elements in the
   * inner loop are H_ * sizeof(DATA_TYPE) bytes apart.
   *
   * Transformation:
   *   - We switch to row-major iteration: for (i) for (j).
   *   - We keep three per-column state arrays:
   *       tm1_col[j] ~ previous imgOut value in column j
   *       ym1_col[j] ~ previous y1 in column j
   *       ym2_col[j] ~ value of y1 two rows above in column j
   *   - For each (i,j) we use these states exactly as the original
   *     recurrence did; then we update them.
   *
   * This preserves numerical semantics but improves cache locality and
   * enables vectorisation of the inner loop over j.
   * ------------------------------------------------------------------*/
  DATA_TYPE tm1_col[H_];
  DATA_TYPE ym1_col[H_];
  DATA_TYPE ym2_col[H_];

  /* Initial conditions: match original tm1 = ym1 = ym2 = 0 per column. */
  for (int j = 0; j < H_; j++) {
    tm1_col[j] = SCALAR_VAL(0.0);
    ym1_col[j] = SCALAR_VAL(0.0);
    ym2_col[j] = SCALAR_VAL(0.0);
  }

  for (int i = 0; i < W_; i++) {
    const DATA_TYPE * restrict out_row = imgOutR[i];
    DATA_TYPE * restrict y1_row = y1R[i];

    for (int j = 0; j < H_; j++) {
      const DATA_TYPE prev_tm1 = tm1_col[j];
      const DATA_TYPE prev_ym1 = ym1_col[j];
      const DATA_TYPE prev_ym2 = ym2_col[j];

      const DATA_TYPE x = out_row[j];
      const DATA_TYPE y = a5 * x + a6 * prev_tm1 + b1 * prev_ym1 + b2 * prev_ym2;

      y1_row[j] = y;

      /* Update column-wise state for next row. */
      tm1_col[j] = x;
      ym2_col[j] = prev_ym1;
      ym1_col[j] = y;
    }
  }

  /* --------------------------------------------------------------------
   * 5. Vertical recursive filter (reverse direction)
   *
   * Same restructuring as step 4, but processing rows bottom-to-top.
   * For each column j we track:
   *   tp1_col[j], tp2_col[j], yp1_col[j], yp2_col[j]
   * mirroring the original scalar recurrence variables.
   * ------------------------------------------------------------------*/
  DATA_TYPE tp1_col[H_];
  DATA_TYPE tp2_col[H_];
  DATA_TYPE yp1_col[H_];
  DATA_TYPE yp2_col[H_];

  for (int j = 0; j < H_; j++) {
    tp1_col[j] = SCALAR_VAL(0.0);
    tp2_col[j] = SCALAR_VAL(0.0);
    yp1_col[j] = SCALAR_VAL(0.0);
    yp2_col[j] = SCALAR_VAL(0.0);
  }

  for (int i = W_ - 1; i >= 0; i--) {
    const DATA_TYPE * restrict out_row = imgOutR[i];
    DATA_TYPE * restrict y2_row = y2R[i];

    for (int j = 0; j < H_; j++) {
      const DATA_TYPE prev_tp1 = tp1_col[j];
      const DATA_TYPE prev_tp2 = tp2_col[j];
      const DATA_TYPE prev_yp1 = yp1_col[j];
      const DATA_TYPE prev_yp2 = yp2_col[j];

      const DATA_TYPE y =
          a7 * prev_tp1 + a8 * prev_tp2 + b1 * prev_yp1 + b2 * prev_yp2;
      y2_row[j] = y;

      /* Update column-wise state for the next (upper) row. */
      tp2_col[j] = prev_tp1;
      tp1_col[j] = out_row[j];
      yp2_col[j] = prev_yp1;
      yp1_col[j] = y;
    }
  }

  /* --------------------------------------------------------------------
   * 6. Final vertical combination: element-wise, trivially parallel.
   * ------------------------------------------------------------------*/
#pragma omp parallel for if (total_size >= DERICHE_OMP_MIN_SIZE) schedule(static)
  for (int i = 0; i < W_; i++) {
    const DATA_TYPE * restrict y1_row = y1R[i];
    const DATA_TYPE * restrict y2_row = y2R[i];
    DATA_TYPE * restrict out_row = imgOutR[i];

    for (int j = 0; j < H_; j++)
      out_row[j] = c2 * (y1_row[j] + y2_row[j]);
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