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
 *  - Precompute and reuse exponential/power terms.
 *  - Introduce local restrict-qualified aliases for better
 *    alias analysis and vectorization.
 *  - Fuse some passes to reduce full-image traversals.
 *  - Improve memory access patterns using row/column-wise
 *    pointers and linear indices.
 *  - Expose row/column-level parallelism with OpenMP (optional).
 *
 * All transformations preserve the mathematical computation
 * performed for each pixel.
 */
static
void kernel_deriche[[gnu::flatten, gnu::noinline]](int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i, j;

    /* Create local restrict-qualified aliases.
     * POLYBENCH_2D uses VLAs; here we restate them with 'restrict'
     * so the compiler can assume no aliasing between arrays and
     * generate better code. We use only these aliases below.
     */
    DATA_TYPE (* restrict imgIn_)[h]  = imgIn;
    DATA_TYPE (* restrict imgOut_)[h] = imgOut;
    DATA_TYPE (* restrict y1_)[h]     = y1;
    DATA_TYPE (* restrict y2_)[h]     = y2;

#pragma scop
    /* Use PolyBench-provided loop bounds, which may differ from
     * raw sizes when benchmarking (e.g., for reduced problem sizes).
     */
    const int w_bound = _PB_W;
    const int h_bound = _PB_H;

    /* Precompute constants to avoid redundant transcendental calls. */
    const DATA_TYPE alpha_val = alpha;
    const DATA_TYPE one       = SCALAR_VAL(1.0);
    const DATA_TYPE two       = SCALAR_VAL(2.0);

    const DATA_TYPE exp_m_alpha  = EXP_FUN(-alpha_val);
    const DATA_TYPE exp_m_2alpha = EXP_FUN(-two * alpha_val);
    const DATA_TYPE exp_p_2alpha = EXP_FUN( two * alpha_val);
    const DATA_TYPE pow_2_m_alpha = POW_FUN(two, -alpha_val);

    const DATA_TYPE tmp = (one - exp_m_alpha);
    const DATA_TYPE k =
        (tmp * tmp) /
        (one + two * alpha_val * exp_m_alpha - exp_p_2alpha);

    /* Filter coefficients (unchanged mathematically). */
    const DATA_TYPE a1 = k;
    const DATA_TYPE a2 = k * exp_m_alpha * (alpha_val - one);
    const DATA_TYPE a3 = k * exp_m_alpha * (alpha_val + one);
    const DATA_TYPE a4 = -k * exp_m_2alpha;

    const DATA_TYPE a5 = a1;
    const DATA_TYPE a6 = a2;
    const DATA_TYPE a7 = a3;
    const DATA_TYPE a8 = a4;

    const DATA_TYPE b1 = pow_2_m_alpha;
    const DATA_TYPE b2 = -exp_m_2alpha;

    const DATA_TYPE c1 = one;
    const DATA_TYPE c2 = one;

    /* ------------------------------------------------------------------ */
    /* Horizontal stage: filter each row with two 1D recursive passes
     * (left-to-right and right-to-left), then combine them.
     *
     * Transformation:
     *  - Original code used three separate loops over rows:
     *      1) forward pass (y1)
     *      2) backward pass (y2)
     *      3) combination into imgOut
     *    Here we fuse (2) and (3) per row to reduce one full-image
     *    traversal and improve row-locality. For each pixel, the
     *    operations and recurrences are identical.
     *
     *  - The outer loop over 'i' (rows) is parallelized: each row
     *    is independent because recurrences are only along 'j'.
     */
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < w_bound; ++i) {
        DATA_TYPE * restrict imgIn_row  = imgIn_[i];
        DATA_TYPE * restrict y1_row     = y1_[i];
        DATA_TYPE * restrict y2_row     = y2_[i];
        DATA_TYPE * restrict imgOut_row = imgOut_[i];

        /* Left-to-right recursive pass on row i. */
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);

        for (j = 0; j < h_bound; ++j) {
            const DATA_TYPE x = imgIn_row[j];
            const DATA_TYPE y = a1 * x + a2 * xm1 + b1 * ym1 + b2 * ym2;
            y1_row[j] = y;
            xm1 = x;
            ym2 = ym1;
            ym1 = y;
        }

        /* Right-to-left recursive pass on row i + horizontal combination.
         * This reproduces the original recurrence for y2 and then
         * immediately computes imgOut[i][j] = c1 * (y1 + y2).
         */
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);

        for (j = h_bound - 1; j >= 0; --j) {
            const DATA_TYPE x = imgIn_row[j];
            const DATA_TYPE y = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;

            y2_row[j] = y; /* same as original y2[i][j] */

            xp2 = xp1;
            xp1 = x;
            yp2 = yp1;
            yp1 = y;

            imgOut_row[j] = c1 * (y1_row[j] + y);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Vertical stage: filter columns with two 1D recursive passes
     * (top-to-bottom and bottom-to-top), then combine them.
     *
     * Transformation:
     *  - Original code used three passes:
     *      4) forward column pass writing y1
     *      5) backward column pass writing y2
     *      6) combination into imgOut
     *    Here, for each column j, we:
     *      - perform the forward pass (y1),
     *      - perform the backward pass (y2),
     *      - immediately combine into imgOut.
     *
     *    This reduces one extra traversal over y1/y2/imgOut and
     *    preserves the exact recurrence structure.
     *
     *  - Accesses are organized through a flattened base pointer and an
     *    index 'idx = i*h + j'. This avoids expensive 2D indexing while
     *    still respecting C's pointer arithmetic rules.
     *
     *  - The outer loop over 'j' (columns) is parallelized. Each column
     *    is independent because recurrences are only along 'i'.
     */
    {
        DATA_TYPE * restrict imgOut_data = &imgOut_[0][0];
        DATA_TYPE * restrict y1_data     = &y1_[0][0];
        DATA_TYPE * restrict y2_data     = &y2_[0][0];
        const int stride = h; /* physical row stride in elements */

#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
        for (j = 0; j < h_bound; ++j) {

            /* Top-to-bottom recursive pass on column j (y1). */
            DATA_TYPE tm1 = SCALAR_VAL(0.0);
            DATA_TYPE ym1 = SCALAR_VAL(0.0);
            DATA_TYPE ym2 = SCALAR_VAL(0.0);

            int idx = j; /* linear index for element [0][j] */
            for (i = 0; i < w_bound; ++i, idx += stride) {
                const DATA_TYPE x = imgOut_data[idx];
                const DATA_TYPE y = a5 * x + a6 * tm1 + b1 * ym1 + b2 * ym2;
                y1_data[idx] = y;
                tm1 = x;
                ym2 = ym1;
                ym1 = y;
            }

            /* Bottom-to-top recursive pass on column j (y2) +
             * final vertical combination into imgOut.
             */
            DATA_TYPE tp1 = SCALAR_VAL(0.0);
            DATA_TYPE tp2 = SCALAR_VAL(0.0);
            DATA_TYPE yp1 = SCALAR_VAL(0.0);
            DATA_TYPE yp2 = SCALAR_VAL(0.0);

            idx = (w_bound - 1) * stride + j; /* element [w_bound-1][j] */
            for (i = w_bound - 1; i >= 0; --i, idx -= stride) {
                const DATA_TYPE x = imgOut_data[idx];
                const DATA_TYPE y = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;

                y2_data[idx] = y; /* same as original y2[i][j] */

                tp2 = tp1;
                tp1 = x;
                yp2 = yp1;
                yp1 = y;

                imgOut_data[idx] = c2 * (y1_data[idx] + y);
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