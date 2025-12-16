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
from exo.libs.memories import DRAM
from exo.core.extern import Extern, _EErr


class _Sqrt(Extern):
    def __init__(self):
        # Name is only for pretty-printing; C code comes from compile()
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
        # Use PolyBench's SQRT_FUN macro for correct precision
        return f"SQRT_FUN(({prim_type}){args[0]})"

    def globl(self, prim_type):
        # SQRT_FUN is provided by the PolyBench headers, but we may still need math.h
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.sqrt(args[0])


sqrt_exo = _Sqrt()


class _ClampStddev(Extern):
    """
    Implements: x <= eps ? 1.0 : x
    Used to avoid value-dependent conditionals in Exo object code.
    """
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
        # Result has the same type as x
        return x_t

    def compile(self, args, prim_type):
        # Use SCALAR_VAL(1.0) to match PolyBench's DATA_TYPE precisely
        return f"(({args[0]} <= {args[1]}) ? SCALAR_VAL(1.0) : {args[0]})"

    def globl(self, prim_type):
        # SCALAR_VAL is defined in the PolyBench headers
        return ""

    def interpret(self, args):
        x, eps = args
        return 1.0 if x <= eps else x


clamp_stddev = _ClampStddev()


@proc
def kernel_correlation(
    m: size,
    n: size,
    float_n: DATA_TYPE,
    data: DATA_TYPE[n, m] @ DRAM,
    corr: DATA_TYPE[m, m] @ DRAM,
    mean: DATA_TYPE[m] @ DRAM,
    stddev: DATA_TYPE[m] @ DRAM,
):
    eps: DATA_TYPE

    # eps = 0.1 (same as SCALAR_VAL(0.1) in the original C code)
    eps = 0.1

    # Compute mean of each column: mean[j] = (1/n) * sum_i data[i,j]
    for j in seq(0, m):
        mean[j] = 0.0
        for i in seq(0, n):
            mean[j] += data[i, j]
        mean[j] = mean[j] / float_n

    # Compute standard deviation of each column
    for j in seq(0, m):
        stddev[j] = 0.0
        for i in seq(0, n):
            stddev[j] += (data[i, j] - mean[j]) * (data[i, j] - mean[j])
        stddev[j] = stddev[j] / float_n
        stddev[j] = sqrt_exo(stddev[j])
        # Handle near-zero stddev values without explicit value-dependent control flow
        stddev[j] = clamp_stddev(stddev[j], eps)

    # Center and reduce the column vectors
    for i in seq(0, n):
        for j in seq(0, m):
            data[i, j] = data[i, j] - mean[j]
            data[i, j] = data[i, j] / (sqrt_exo(float_n) * stddev[j])

    # Compute the m x m correlation matrix
    for i in seq(0, m - 1):
        corr[i, i] = 1.0
        for j in seq(i + 1, m):
            corr[i, j] = 0.0
            for k in seq(0, n):
                corr[i, j] += data[k, i] * data[k, j]
            corr[j, i] = corr[i, j]
    corr[m - 1, m - 1] = 1.0
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