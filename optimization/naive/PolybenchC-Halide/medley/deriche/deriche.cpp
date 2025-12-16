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
 *
 * We express the initialization as Halide Funcs and give them
 * a cache‑friendly, vectorized schedule so that large inputs
 * are initialized efficiently. All initialization is done
 * before we start timing the main kernel.
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

    // In C: imgIn has type [W][H] and is accessed as imgIn[i][j].
    // The second index j is the fastest varying. In Halide::Buffer,
    // the first dimension is the fastest varying, so we map:
    //
    //   C:       imgIn[i][j]
    //   Halide:  imgIn(j, i)
    //
    // and allocate Buffer(h, w).
    Var i("i"), j("j");
    Func init_in("init_in");

    // imgIn[i][j] = ((313*i + 991*j) % 65536) / 65535.0f;
    // Remember to map indices: C's (i,j) -> (j,i) here.
    Expr val = (313 * i + 991 * j) % 65536;
    init_in(j, i) = cast<num_t>(val) / cast<num_t>(65535.0f);

    // imgOut is not initialized in the original C code, but we'll
    // set it to zero for completeness (and deterministic output).
    Func init_out("init_out");
    init_out(j, i) = cast<num_t>(0);

    // -----------------------------------------------------------------
    // Schedule for initialization:
    //
    //  - j is the innermost (contiguous) dimension.
    //  - Vectorize across j using the natural vector width.
    //  - Parallelize across i (rows) to use multiple cores.
    // -----------------------------------------------------------------
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    init_in
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    init_out
        .compute_root()
        .parallel(i)
        .vectorize(j, vec_width);

    // Fill imgIn and imgOut.
    init_in.realize(imgIn);
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
    // Halide pipeline for the non‑recurrent (purely pointwise) parts
    // of the kernel: the two (y1 + y2) combinations.
    //
    // We express the combination once as a Halide Func parameterized
    // by a scalar Param "c_param", and realize it twice with different
    // values (c1 and c2). This avoids redundant compiled code and lets
    // us give a high‑quality schedule for this large, embarrassingly
    // parallel stage.
    // --------------------------------------------------------------------
    Var x("x"), y("y");

    // y1 and y2 will be passed into Halide as ImageParams.
    ImageParam y1_param(type_of<num_t>(), 2, "y1_param");
    ImageParam y2_param(type_of<num_t>(), 2, "y2_param");

    // Scalar factor for the combination (c1 or c2).
    Param<num_t> c_param("c_param");

    // Single combination Func:
    //   imgOut = c_param * (y1 + y2)
    Func combine("combine");
    combine(x, y) = c_param * (y1_param(x, y) + y2_param(x, y));

    // --------------------------------------------------------------------
    // Schedule for the combination:
    //
    //  - x corresponds to the contiguous dimension (dim 0 of the
    //    underlying Buffer: index pattern (x, y) = (j, i)).
    //  - We vectorize across x.
    //  - We tile y into strips and parallelize across those strips.
    // --------------------------------------------------------------------
    Target target = get_host_target();
    const int vec_width = target.natural_vector_size(type_of<num_t>());

    Var yo("yo"), yi("yi");
    combine
        .compute_root()
        // Process rows in parallel in tiles of 32 scanlines.
        .split(y, yo, yi, 32)
        .parallel(yo)
        // Vectorize across contiguous x.
        .vectorize(x, vec_width);

    // JIT‑compile the combination pipeline in advance so we don't
    // measure compilation time inside the timed region.
    combine.compile_jit(target);

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

    // c1 and c2 are both 1.0 in the Polybench spec, but we keep them
    // explicit in case they are changed in deriche.hpp for tuning.
    num_t c1 = (num_t)1;
    num_t c2 = (num_t)1;

    // --------------------------------------------------------------------
    // Main computational kernel: Deriche filter.
    //
    // The core of Deriche is a set of 1D IIR recurrences along rows
    // and columns. These recurrences create loop‑carried dependencies
    // that Halide cannot legally express as a single pure Func.
    //
    // Following the guidelines for such dynamic‑programming patterns,
    // we implement the recurrences as explicit C++ loops over
    // Halide::Buffer data, and use Halide Funcs for the purely
    // pointwise passes (the two combinations).
    //
    // Buffer indexing:
    //   original C:   imgIn[i][j]
    //   Halide here:  imgIn(j, i)
    // where i in [0, w) and j in [0, h).
    //
    // We keep the original mapping and printing semantics, but take
    // care to traverse the fast dimension (j) as the inner loop for
    // the horizontal passes, and use a good per‑row / per‑column
    // access pattern for cache locality.
    // --------------------------------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    // 1) Horizontal forward recursion (left -> right).
    //
    // For each row i, we sweep across j in the inner loop. Because
    // dimension 0 (j) has unit stride, this is a contiguous traversal
    // in memory and plays well with the cache hierarchy.
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
    //
    // Same row‑wise access order, but sweeping j from right to left.
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
    //
    // This is the first place where we use the Halide pipeline
    // "combine". It is fully vectorized and parallelized across rows.
    y1_param.set(y1);
    y2_param.set(y2);
    c_param.set(c1);
    combine.realize(imgOut);

    // 4) Vertical forward recursion (top -> bottom).
    //
    // Here the recurrence runs along the i dimension (original C's
    // outer loop), which corresponds to the *second* dimension of our
    // Buffers. This means accesses are strided in memory. We keep the
    // algorithmic order to preserve correctness; modern CPUs can
    // still handle such strided streaming loops reasonably well via
    // hardware prefetching.
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
    //
    // Again we use the Halide "combine" Func with a different scalar
    // factor. All the scheduling goodness (vectorization, parallelism)
    // is reused automatically.
    y1_param.set(y1);
    y2_param.set(y2);
    c_param.set(c2);
    combine.realize(imgOut);

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