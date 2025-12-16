cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(
  NoarrPolybench VERSION 4.2.1
  DESCRIPTION "Polybench kernels implemented using Noarr Structures"
  LANGUAGES CXX
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# enable openmp if available
add_compile_options(-fopenmp -pthread -march=native -mtune=native -Wfatal-errors)

link_libraries(gomp)
link_libraries(m)

if(DEFINED ENV{NOARR_STRUCTURES_DIR})
    message(STATUS "Using NOARR_STRUCTURES_DIR from environment: $ENV{NOARR_STRUCTURES_DIR}")
    include_directories($ENV{NOARR_STRUCTURES_DIR}/include)
else()
    include_directories(${CMAKE_SOURCE_DIR}/../submodules/noarr-structures/include)
endif()

# datamining
add_executable(correlation datamining/correlation/correlation.cpp)
add_executable(covariance datamining/covariance/covariance.cpp)

# linear algebra - blas
add_executable(gemm linear-algebra/blas/gemm/gemm.cpp)
add_executable(gemver linear-algebra/blas/gemver/gemver.cpp)
add_executable(gesummv linear-algebra/blas/gesummv/gesummv.cpp)
add_executable(symm linear-algebra/blas/symm/symm.cpp)
add_executable(syr2k linear-algebra/blas/syr2k/syr2k.cpp)
add_executable(syrk linear-algebra/blas/syrk/syrk.cpp)
add_executable(trmm linear-algebra/blas/trmm/trmm.cpp)

# linear algebra - kernels
add_executable(2mm linear-algebra/kernels/2mm/2mm.cpp)
add_executable(3mm linear-algebra/kernels/3mm/3mm.cpp)
add_executable(atax linear-algebra/kernels/atax/atax.cpp)
add_executable(bicg linear-algebra/kernels/bicg/bicg.cpp)
add_executable(doitgen linear-algebra/kernels/doitgen/doitgen.cpp)
add_executable(mvt linear-algebra/kernels/mvt/mvt.cpp)

# linear algebra - solvers
add_executable(cholesky linear-algebra/solvers/cholesky/cholesky.cpp)
add_executable(durbin linear-algebra/solvers/durbin/durbin.cpp)
add_executable(gramschmidt linear-algebra/solvers/gramschmidt/gramschmidt.cpp)
add_executable(lu linear-algebra/solvers/lu/lu.cpp)
add_executable(ludcmp linear-algebra/solvers/ludcmp/ludcmp.cpp)
add_executable(trisolv linear-algebra/solvers/trisolv/trisolv.cpp)

# medley
add_executable(deriche medley/deriche/deriche.cpp)
add_executable(floyd-warshall medley/floyd-warshall/floyd-warshall.cpp)
add_executable(nussinov medley/nussinov/nussinov.cpp)

# stencils
add_executable(adi stencils/adi/adi.cpp)
add_executable(fdtd-2d stencils/fdtd-2d/fdtd-2d.cpp)
add_executable(heat-3d stencils/heat-3d/heat-3d.cpp)
add_executable(jacobi-1d stencils/jacobi-1d/jacobi-1d.cpp)
add_executable(jacobi-2d stencils/jacobi-2d/jacobi-2d.cpp)
add_executable(seidel-2d stencils/seidel-2d/seidel-2d.cpp)

if (MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()
