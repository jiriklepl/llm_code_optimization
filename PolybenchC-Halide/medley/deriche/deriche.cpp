#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "Halide.h"

// include common definitions (DATA_TYPE, W, H, etc)
#include "defines.hpp"

// include benchmark-specific definitions (if any)
#include "deriche.hpp"

using num_t = DATA_TYPE;

using namespace Halide;

/*
 * Array initialization: replicate the Polybench init_array,
 * but using Halide::Buffer for the arrays.
 */
static void init_array(int w,
                       int h,
                       num_t *alpha,
                       Halide::Buffer<num_t, 2> imgIn,
                       Halide::Buffer<num_t, 2> imgOut) {
    // Silence unused parameter warnings (sizes are implied by the Buffers).
    (void)w;
    (void)h;

    // Parameter of the filter.
    *alpha = (num_t)0.25;

    // imgIn[i][j] = ((313*i + 991*j) % 65536) / 65535.0f;
    //
    // In C: imgIn has type [W][H] and is accessed as imgIn[i][j].
    // The second index j is the fastest varying. In Halide::Buffer,
    // the first dimension is the fastest varying, so we map:
    //   imgIn(i_c, j_c)  <->  imgIn(j_c, i_c)
    // and allocate Buffer(h, w).
    Var i("i"), j("j");
    Func init_in("init_in");

    Expr val = (313 * i + 991 * j) % 65536;
    init_in(j, i) = cast<num_t>(val) / cast<num_t>(65535.0f);

    // Fill imgIn.
    init_in.realize(imgIn);

    // imgOut is not initialized in the original C code, but we'll
    // set it to zero for completeness.
    Func init_out("init_out");
    init_out(j, i) = cast<num_t>(0);
    init_out.realize(imgOut);
}

int main(int argc, char *argv[]) {
    // Problem size (Polybench constants).
    int w = W;
    int h = H;

    // Allocate arrays as Halide buffers.
    //
    // C layout: imgIn[w][h]   -> indices imgIn[i][j]
    // Halide   : Buffer(h, w) -> indices imgIn(j, i)
    Halide::Buffer<num_t, 2> imgIn(h, w);
    Halide::Buffer<num_t, 2> imgOut(h, w);
    Halide::Buffer<num_t, 2> y1(h, w);
    Halide::Buffer<num_t, 2> y2(h, w);

    // Scalar parameter.
    num_t alpha;

    // Initialize input.
    init_array(w, h, &alpha, imgIn, imgOut);

    // --------------------------------------------------------------------
    // Halide pipeline pieces for the non‑recurrent parts of the kernel
    // (the two pointwise combinations).
    // --------------------------------------------------------------------
    Var x("x"), y("y");

    // y1 and y2 will be passed into Halide as ImageParams.
    ImageParam y1_param(type_of<num_t>(), 2, "y1_param");
    ImageParam y2_param(type_of<num_t>(), 2, "y2_param");

    // First combination: imgOut = c1 * (y1 + y2)
    Func combine1("combine1");
    // Second combination: imgOut = c2 * (y1 + y2)
    Func combine2("combine2");

    num_t c1 = (num_t)1;
    num_t c2 = (num_t)1;

    combine1(x, y) = c1 * (y1_param(x, y) + y2_param(x, y));
    combine2(x, y) = c2 * (y1_param(x, y) + y2_param(x, y));

    // Simple row-major traversal (x = column / fast dimension).
    combine1.reorder(x, y);
    combine2.reorder(x, y);

    // JIT-compile the small Halide pipelines in advance so we don't
    // measure compilation time.
    combine1.compile_jit();
    combine2.compile_jit();

    // --------------------------------------------------------------------
    // Compute Deriche filter coefficients on the host (scalar math).
    // This mirrors the Polybench kernel_deriche coefficient setup.
    // --------------------------------------------------------------------
    num_t ea   = std::exp(-alpha);                 // exp(-alpha)
    num_t e2a  = std::exp(num_t(2.0) * alpha);     // exp(2*alpha)
    num_t e_2a = std::exp(num_t(-2.0) * alpha);    // exp(-2*alpha)

    num_t k = (num_t(1.0) - ea) * (num_t(1.0) - ea) /
              (num_t(1.0) + num_t(2.0) * alpha * ea - e2a);

    num_t a1 = k;
    num_t a5 = k;

    num_t a2 = k * ea * (alpha - num_t(1.0));
    num_t a6 = a2;

    num_t a3 = k * ea * (alpha + num_t(1.0));
    num_t a7 = a3;

    num_t a4 = -k * e_2a;
    num_t a8 = a4;

    num_t b1 = std::pow(num_t(2.0), -alpha);
    num_t b2 = -e_2a;

    // --------------------------------------------------------------------
    // Main computational kernel: Deriche filter.
    //
    // Due to the loop-carried dependencies (IIR recurrences) along rows
    // and columns, we implement the recursive passes as explicit C++
    // loops over Halide::Buffer data, and use Halide Funcs for the
    // purely pointwise passes (the two combinations).
    //
    // Buffer indexing:
    //   original C:   imgIn[i][j]
    //   Halide here:  imgIn(j, i)
    // where i in [0, w) and j in [0, h).
    // --------------------------------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    // 1) Horizontal forward recursion (left -> right).
    for (int i = 0; i < w; ++i) {
        num_t ym1 = num_t(0);
        num_t ym2 = num_t(0);
        num_t xm1 = num_t(0);

        for (int j = 0; j < h; ++j) {
            num_t x_ij = imgIn(j, i);
            num_t y_ij = a1 * x_ij + a2 * xm1 + b1 * ym1 + b2 * ym2;
            y1(j, i) = y_ij;

            xm1 = x_ij;
            ym2 = ym1;
            ym1 = y_ij;
        }
    }

    // 2) Horizontal backward recursion (right -> left).
    for (int i = 0; i < w; ++i) {
        num_t yp1 = num_t(0);
        num_t yp2 = num_t(0);
        num_t xp1 = num_t(0);
        num_t xp2 = num_t(0);

        for (int j = h - 1; j >= 0; --j) {
            num_t y_ij = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            y2(j, i) = y_ij;

            xp2 = xp1;
            xp1 = imgIn(j, i);
            yp2 = yp1;
            yp1 = y_ij;
        }
    }

    // 3) First pointwise combination:
    //    imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
    y1_param.set(y1);
    y2_param.set(y2);
    combine1.realize(imgOut);

    // 4) Vertical forward recursion (top -> bottom).
    for (int j = 0; j < h; ++j) {
        num_t tm1 = num_t(0);
        num_t ym1 = num_t(0);
        num_t ym2 = num_t(0);

        for (int i = 0; i < w; ++i) {
            num_t x_ij = imgOut(j, i);
            num_t y_ij = a5 * x_ij + a6 * tm1 + b1 * ym1 + b2 * ym2;
            y1(j, i) = y_ij;

            tm1 = x_ij;
            ym2 = ym1;
            ym1 = y_ij;
        }
    }

    // 5) Vertical backward recursion (bottom -> top).
    for (int j = 0; j < h; ++j) {
        num_t tp1 = num_t(0);
        num_t tp2 = num_t(0);
        num_t yp1 = num_t(0);
        num_t yp2 = num_t(0);

        for (int i = w - 1; i >= 0; --i) {
            num_t y_ij = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            y2(j, i) = y_ij;

            tp2 = tp1;
            tp1 = imgOut(j, i);
            yp2 = yp1;
            yp1 = y_ij;
        }
    }

    // 6) Final pointwise combination:
    //    imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
    y1_param.set(y1);
    y2_param.set(y2);
    combine2.realize(imgOut);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<long double>(end - start);

    // --------------------------------------------------------------------
    // Output and timing, similar style to example/gemm.cpp.
    // --------------------------------------------------------------------
    using namespace std::string_literals;

    if (argc > 0 && argv[0] != ""s) {
        std::cerr << std::fixed << std::setprecision(6);
        for (int i = 0; i < w; ++i) {
            for (int j = 0; j < h; ++j) {
                // Original C prints imgOut[i][j]; here we map to imgOut(j, i).
                std::cerr << imgOut(j, i) << '\n';
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << duration.count() << std::endl;

    return 0;
}