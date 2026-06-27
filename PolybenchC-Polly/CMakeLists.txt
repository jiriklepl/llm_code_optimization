cmake_minimum_required(VERSION 3.10)

# set the project name
project(PolybenchC VERSION 4.2.1 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)

set(POLYBENCH_DATASET_SIZE "LARGE" CACHE STRING "PolyBench dataset size")
set_property(
    CACHE POLYBENCH_DATASET_SIZE
    PROPERTY STRINGS MINI SMALL MEDIUM LARGE EXTRALARGE
)
set(POLYBENCH_DATA_TYPE "FLOAT" CACHE STRING "PolyBench data type")
set_property(
    CACHE POLYBENCH_DATA_TYPE
    PROPERTY STRINGS FLOAT DOUBLE INT
)

set(POLYBENCH_DATASET_SIZES MINI SMALL MEDIUM LARGE EXTRALARGE)
if(NOT POLYBENCH_DATASET_SIZE IN_LIST POLYBENCH_DATASET_SIZES)
    message(FATAL_ERROR "Unsupported PolyBench dataset size: ${POLYBENCH_DATASET_SIZE}")
endif()

set(POLYBENCH_DATA_TYPES FLOAT DOUBLE INT)
if(NOT POLYBENCH_DATA_TYPE IN_LIST POLYBENCH_DATA_TYPES)
    message(FATAL_ERROR "Unsupported PolyBench data type: ${POLYBENCH_DATA_TYPE}")
endif()

include_directories(utilities)

find_package(OpenMP REQUIRED COMPONENTS C)

add_compile_definitions(
    POLYBENCH_TIME
    POLYBENCH_DUMP_ARRAYS
    ${POLYBENCH_DATASET_SIZE}_DATASET
    DATA_TYPE_IS_${POLYBENCH_DATA_TYPE}
    _POSIX_C_SOURCE=200809L
)
add_compile_options(
    -pthread
    -march=native
    -mtune=native
    -Wfatal-errors
    "SHELL:-mllvm -polly"
    "SHELL:-mllvm -polly-vectorizer=stripmine"
    "SHELL:-mllvm -polly-parallel"
)

link_libraries(m)
link_libraries(OpenMP::OpenMP_C)

add_executable(covariance datamining/covariance/covariance.c utilities/polybench.c)
add_executable(correlation datamining/correlation/correlation.c utilities/polybench.c)
add_executable(lu linear-algebra/solvers/lu/lu.c utilities/polybench.c)
add_executable(cholesky linear-algebra/solvers/cholesky/cholesky.c utilities/polybench.c)
add_executable(gramschmidt linear-algebra/solvers/gramschmidt/gramschmidt.c utilities/polybench.c)
add_executable(durbin linear-algebra/solvers/durbin/durbin.c utilities/polybench.c)
add_executable(trisolv linear-algebra/solvers/trisolv/trisolv.c utilities/polybench.c)
add_executable(ludcmp linear-algebra/solvers/ludcmp/ludcmp.c utilities/polybench.c)
add_executable(syr2k linear-algebra/blas/syr2k/syr2k.c utilities/polybench.c)
add_executable(gemm linear-algebra/blas/gemm/gemm.c utilities/polybench.c)
add_executable(gemver linear-algebra/blas/gemver/gemver.c utilities/polybench.c)
add_executable(syrk linear-algebra/blas/syrk/syrk.c utilities/polybench.c)
add_executable(gesummv linear-algebra/blas/gesummv/gesummv.c utilities/polybench.c)
add_executable(trmm linear-algebra/blas/trmm/trmm.c utilities/polybench.c)
add_executable(symm linear-algebra/blas/symm/symm.c utilities/polybench.c)
add_executable(bicg linear-algebra/kernels/bicg/bicg.c utilities/polybench.c)
add_executable(mvt linear-algebra/kernels/mvt/mvt.c utilities/polybench.c)
add_executable(atax linear-algebra/kernels/atax/atax.c utilities/polybench.c)
add_executable(3mm linear-algebra/kernels/3mm/3mm.c utilities/polybench.c)
add_executable(doitgen linear-algebra/kernels/doitgen/doitgen.c utilities/polybench.c)
add_executable(2mm linear-algebra/kernels/2mm/2mm.c utilities/polybench.c)
add_executable(nussinov medley/nussinov/nussinov.c utilities/polybench.c)
add_executable(floyd-warshall medley/floyd-warshall/floyd-warshall.c utilities/polybench.c)
add_executable(deriche medley/deriche/deriche.c utilities/polybench.c)
add_executable(jacobi-1d stencils/jacobi-1d/jacobi-1d.c utilities/polybench.c)
add_executable(fdtd-2d stencils/fdtd-2d/fdtd-2d.c utilities/polybench.c)
add_executable(jacobi-2d stencils/jacobi-2d/jacobi-2d.c utilities/polybench.c)
add_executable(seidel-2d stencils/seidel-2d/seidel-2d.c utilities/polybench.c)
add_executable(heat-3d stencils/heat-3d/heat-3d.c utilities/polybench.c)
add_executable(adi stencils/adi/adi.c utilities/polybench.c)
