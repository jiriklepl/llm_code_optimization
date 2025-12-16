/**
 * Exo correlation driver: mirrors PolyBench/C correlation.c but calls the Exo-generated kernel.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"

/* Include the Exo-generated kernel header. */
#include "generated/correlation/correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}

/* Kernel implementation in Exo:
EXO START
from __future__ import annotations

from exo import *
from exo.API_scheduling import *  # scheduling primitives (simplify, rename, etc.)
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


# Extern for sqrt that delegates to PolyBench's SQRT_FUN macro so that
# the precision (float/double) matches DATA_TYPE exactly.
class _Sqrt(Extern):
    def __init__(self):
        super().__init__("sqrt_exo")

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
        # SQRT_FUN is defined by the PolyBench headers.
        return f"SQRT_FUN(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # math.h is required for sqrt; SQRT_FUN itself comes from headers.
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_exo = _Sqrt()


# Extern implementing the stddev clamp without value-dependent control
# flow in the Exo object code:
#   clamp_stddev(x, eps) ≡ (x <= eps ? 1.0 : x)
class _ClampStddev(Extern):
    # Implements: x <= eps ? 1.0 : x
    def __init__(self):
        super().__init__("clamp_stddev")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")
        x_t = args[0].type
        eps_t = args[1].type
        if not (x_t.is_real_scalar() and eps_t.is_real_scalar()):
            raise _EErr(
                f"expected real scalar arguments, but got types {x_t} and {eps_t}"
            )
        return x_t

    def compile(self, args, prim_type):
        # SCALAR_VAL is the PolyBench macro for a literal DATA_TYPE value.
        return f"(({args[0]} <= {args[1]}) ? SCALAR_VAL(1.0) : {args[0]})"

    def globl(self, prim_type):
        # SCALAR_VAL is provided by PolyBench headers; nothing extra needed.
        return ""

    def interpret(self, args):
        x, eps = args
        return 1.0 if x <= eps else x


clamp_stddev = _ClampStddev()


# Baseline-but-optimized correlation kernel.
#
# Key improvements over the naive translation:
#   - All passes over data[n, m] use row-major traversal (i outer, j inner)
#     for better cache locality.
#   - sqrt(float_n) is computed once and reused; division in the inner
#     normalization loop is replaced by a multiply via a per-column scale[j].
#   - The correlation matrix is computed as a sequence of outer products,
#     streaming each normalized row exactly once and updating only the
#     upper triangle, then mirroring to enforce symmetry.
@proc
def kernel_correlation_base(
    m: size,                         # number of columns (variables)
    n: size,                         # number of rows (observations)
    float_n: DATA_TYPE,              # n cast to DATA_TYPE
    data: DATA_TYPE[n, m] @ DRAM,    # input data, normalized in-place
    corr: DATA_TYPE[m, m] @ DRAM,    # output correlation matrix
    mean: DATA_TYPE[m] @ DRAM,       # per-column means
    stddev: DATA_TYPE[m] @ DRAM,     # per-column stddevs (after clamping)
):
    # Small positive threshold for clamping stddev, as in PolyBench.
    eps: DATA_TYPE

    # sqrt(float_n) and its reciprocal, hoisted out of the main loops.
    sqrt_n: DATA_TYPE
    inv_sqrt_n: DATA_TYPE

    # Per-column scaling factor used in normalization:
    #   scale[j] = 1 / (sqrt_n * stddev[j])
    # This turns a division inside the large normalization loop into a
    # multiplication and avoids recomputing sqrt_n there.
    scale: DATA_TYPE[m] @ DRAM

    eps = 0.1

    # Compute sqrt(float_n) once using the same SQRT_FUN-based extern.
    sqrt_n = sqrt_exo(float_n)
    inv_sqrt_n = 1.0 / sqrt_n

    # ---------------------------------------------------------------
    # 1. Column means:
    #      mean[j] = (1/n) * sum_i data[i, j]
    #
    # We first zero mean, then accumulate in row-major order for good
    # spatial locality on data (i outer, j inner), and finally divide.
    # ---------------------------------------------------------------
    for j in seq(0, m):
        mean[j] = 0.0

    for i in seq(0, n):
        for j in seq(0, m):
            mean[j] += data[i, j]

    for j in seq(0, m):
        mean[j] = mean[j] / float_n

    # ---------------------------------------------------------------
    # 2. Column standard deviations:
    #
    #   var_j   = (1/n) * sum_i (data[i, j] - mean[j])^2
    #   stdraw  = sqrt(var_j)
    #   stddev[j] = clamp_stddev(stdraw, eps)
    #
    # stddev[j] is computed exactly as in the PolyBench kernel, but
    # using row-major traversal and the clamp_stddev extern to avoid
    # value-dependent control flow in Exo IR.
    # ---------------------------------------------------------------
    for j in seq(0, m):
        stddev[j] = 0.0

    for i in seq(0, n):
        for j in seq(0, m):
            diff: DATA_TYPE
            diff = data[i, j] - mean[j]
            stddev[j] += diff * diff

    for j in seq(0, m):
        stddev[j] = stddev[j] / float_n
        stddev[j] = sqrt_exo(stddev[j])
        # Implements: stddev[j] = (stddev[j] <= eps ? 1.0 : stddev[j])
        stddev[j] = clamp_stddev(stddev[j], eps)

    # ---------------------------------------------------------------
    # 3. Precompute per-column normalization factors:
    #
    #   scale[j] = 1 / (sqrt(float_n) * stddev[j])
    #
    # This eliminates the per-element sqrt and division from the
    # normalization loop, replacing them with a single multiply per
    # element by scale[j].
    # ---------------------------------------------------------------
    for j in seq(0, m):
        scale[j] = inv_sqrt_n / stddev[j]

    # ---------------------------------------------------------------
    # 4. Center and normalize data in place:
    #
    #   data[i, j] = (data[i, j] - mean[j]) * scale[j]
    #
    # which is algebraically equivalent to the original:
    #
    #   (data[i, j] - mean[j]) / (sqrt(float_n) * stddev[j])
    #
    # but uses only a subtraction and a multiplication in the inner
    # loop, with row-major access (i outer, j inner).
    # ---------------------------------------------------------------
    for i in seq(0, n):
        for j in seq(0, m):
            data[i, j] = (data[i, j] - mean[j]) * scale[j]

    # ---------------------------------------------------------------
    # 5. Correlation matrix as a sum of outer products:
    #
    # After normalization, correlation is:
    #
    #   corr[p, q] = sum_i data[i, p] * data[i, q]
    #
    # The original kernel computes only the upper triangle (p < q),
    # mirrors it, and forces the diagonal to exactly 1.0. We follow
    # the same semantics but reorder loops to:
    #   - stream each normalized row i exactly once,
    #   - access data[i, *] contiguously,
    #   - accumulate only p < q, then symmetrize.
    # ---------------------------------------------------------------

    # Initialize the full matrix to zero.
    for p in seq(0, m):
        for q in seq(0, m):
            corr[p, q] = 0.0

    # Accumulate contributions row by row (outer products).
    for i in seq(0, n):
        for p in seq(0, m):
            tmp_p: DATA_TYPE
            tmp_p = data[i, p]
            # Only update the upper triangle (q > p).
            for q in seq(p + 1, m):
                corr[p, q] += tmp_p * data[i, q]

    # Enforce exact unit diagonal and symmetry, matching PolyBench.
    for p in seq(0, m):
        corr[p, p] = 1.0
        for q in seq(p + 1, m):
            corr[q, p] = corr[p, q]


# Apply a simple scheduling pass (expression simplification) and
# expose the optimized kernel under the name expected by the C driver.
_kernel_opt = simplify(kernel_correlation_base)
kernel_correlation = rename(_kernel_opt, "kernel_correlation")
EXO END
*/


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run Exo kernel. Flatten PolyBench views to 1D pointers. */
  kernel_correlation (/*ctxt=*/NULL, m, n,
                      (DATA_TYPE*)&float_n,
                      (DATA_TYPE*)POLYBENCH_ARRAY(data),
                      (DATA_TYPE*)POLYBENCH_ARRAY(corr),
                      (DATA_TYPE*)POLYBENCH_ARRAY(mean),
                      (DATA_TYPE*)POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}