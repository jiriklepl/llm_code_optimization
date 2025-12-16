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
/* Original code provided by Gael Deest */
static
void kernel_deriche [[gnu::flatten, gnu::noinline]] (int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i, j;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
    /* Precompute constants that depend only on alpha.
     * This removes redundant transcendental calls from the hot loops. */
    const DATA_TYPE one = SCALAR_VAL(1.0);
    const DATA_TYPE two = SCALAR_VAL(2.0);

    const DATA_TYPE ea      = EXP_FUN(-alpha);        /* e^(-alpha)   */
    const DATA_TYPE e2a_pos = EXP_FUN(two * alpha);   /* e^(+2*alpha) */
    const DATA_TYPE e2a_neg = EXP_FUN(-two * alpha);  /* e^(-2*alpha) */

    k  = (one - ea) * (one - ea) /
         (one + two * alpha * ea - e2a_pos);

    a1 = a5 = k;
    a2 = a6 = k * ea * (alpha - one);
    a3 = a7 = k * ea * (alpha + one);
    a4 = a8 = -k * e2a_neg;
    b1 = POW_FUN(two, -alpha);
    b2 = -e2a_neg;
    c1 = c2 = one;

    const int w_bound = _PB_W;
    const int h_bound = _PB_H;

    /* Local restrict-qualified aliases help the compiler reason about the
     * absence of aliasing between the different image buffers, improving
     * optimization of the inner loops. */
    DATA_TYPE (*restrict imgIn_ )[H] = imgIn;
    DATA_TYPE (*restrict imgOut_)[H] = imgOut;
    DATA_TYPE (*restrict y1_    )[H] = y1;
    DATA_TYPE (*restrict y2_    )[H] = y2;

    (void)w; (void)h; /* Parameters kept for interface compatibility. */

    /* --------------------------------------------------------------------
     * First stage: horizontal recursive filtering (per row).
     *
     * We fuse, for each row i:
     *   1) forward pass (left -> right) into y1
     *   2) backward pass (right -> left) into y2
     *   3) combination into imgOut
     *
     * Rows are independent of each other, so this fusion preserves
     * semantics while improving cache locality.
     * ------------------------------------------------------------------ */
    for (i = 0; i < w_bound; i++) {
        DATA_TYPE *restrict imgIn_row  = imgIn_[i];
        DATA_TYPE *restrict y1_row     = y1_[i];
        DATA_TYPE *restrict y2_row     = y2_[i];
        DATA_TYPE *restrict imgOut_row = imgOut_[i];

        /* Forward recursion (left to right). */
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);

        for (j = 0; j < h_bound; j++) {
            const DATA_TYPE x = imgIn_row[j];
            const DATA_TYPE y = a1 * x + a2 * xm1 + b1 * ym1 + b2 * ym2;
            y1_row[j] = y;
            xm1 = x;
            ym2 = ym1;
            ym1 = y;
        }

        /* Backward recursion (right to left). */
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);

        for (j = h_bound - 1; j >= 0; j--) {
            const DATA_TYPE y = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            y2_row[j] = y;
            const DATA_TYPE x = imgIn_row[j];
            xp2 = xp1;
            xp1 = x;
            yp2 = yp1;
            yp1 = y;
        }

        /* Combine the two horizontal passes. */
        for (j = 0; j < h_bound; j++) {
            imgOut_row[j] = c1 * (y1_row[j] + y2_row[j]);
        }
    }

    /* --------------------------------------------------------------------
     * Second stage: vertical recursive filtering (per column).
     *
     * Columns are independent of each other, so we similarly fuse, for
     * each column j:
     *   1) forward pass (top -> bottom) into y1
     *   2) backward pass (bottom -> top) into y2
     *   3) final combination into imgOut
     *
     * This reduces the number of full-image sweeps and improves temporal
     * locality for each column while leaving the per-column recursion
     * order unchanged.
     * ------------------------------------------------------------------ */
    for (j = 0; j < h_bound; j++) {
        /* Forward recursion (top to bottom). */
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);

        for (i = 0; i < w_bound; i++) {
            const DATA_TYPE x = imgOut_[i][j];
            const DATA_TYPE y = a5 * x + a6 * tm1 + b1 * ym1 + b2 * ym2;
            y1_[i][j] = y;
            tm1 = x;
            ym2 = ym1;
            ym1 = y;
        }

        /* Backward recursion (bottom to top). */
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);

        for (i = w_bound - 1; i >= 0; i--) {
            const DATA_TYPE y = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            y2_[i][j] = y;
            const DATA_TYPE x = imgOut_[i][j];
            tp2 = tp1;
            tp1 = x;
            yp2 = yp1;
            yp1 = y;
        }

        /* Final combination for this column. */
        for (i = 0; i < w_bound; i++) {
            imgOut_[i][j] = c2 * (y1_[i][j] + y2_[i][j]);
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