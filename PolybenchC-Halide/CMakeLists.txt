cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(
  HalidePolybench VERSION 4.2.1
  DESCRIPTION "Polybench kernels implemented using Halide"
  LANGUAGES CXX
)

# find Halide
find_package(Halide REQUIRED)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# enable openmp if available
add_compile_options(-fopenmp -pthread -march=native -mtune=native -Wfatal-errors)

link_libraries(gomp)
link_libraries(m)

# datamining
add_executable(correlation datamining/correlation/correlation.cpp)
target_link_libraries(correlation PUBLIC Halide::Halide)
add_executable(covariance datamining/covariance/covariance.cpp)
target_link_libraries(covariance PUBLIC Halide::Halide)

# linear algebra - blas
add_executable(gemm linear-algebra/blas/gemm/gemm.cpp)
target_link_libraries(gemm PUBLIC Halide::Halide)
add_executable(gemver linear-algebra/blas/gemver/gemver.cpp)
target_link_libraries(gemver PUBLIC Halide::Halide)
add_executable(gesummv linear-algebra/blas/gesummv/gesummv.cpp)
target_link_libraries(gesummv PUBLIC Halide::Halide)
add_executable(symm linear-algebra/blas/symm/symm.cpp)
target_link_libraries(symm PUBLIC Halide::Halide)
add_executable(syr2k linear-algebra/blas/syr2k/syr2k.cpp)
target_link_libraries(syr2k PUBLIC Halide::Halide)
add_executable(syrk linear-algebra/blas/syrk/syrk.cpp)
target_link_libraries(syrk PUBLIC Halide::Halide)
add_executable(trmm linear-algebra/blas/trmm/trmm.cpp)
target_link_libraries(trmm PUBLIC Halide::Halide)

# linear algebra - kernels
add_executable(2mm linear-algebra/kernels/2mm/2mm.cpp)
target_link_libraries(2mm PUBLIC Halide::Halide)
add_executable(3mm linear-algebra/kernels/3mm/3mm.cpp)
target_link_libraries(3mm PUBLIC Halide::Halide)
add_executable(atax linear-algebra/kernels/atax/atax.cpp)
target_link_libraries(atax PUBLIC Halide::Halide)
add_executable(bicg linear-algebra/kernels/bicg/bicg.cpp)
target_link_libraries(bicg PUBLIC Halide::Halide)
add_executable(doitgen linear-algebra/kernels/doitgen/doitgen.cpp)
target_link_libraries(doitgen PUBLIC Halide::Halide)
add_executable(mvt linear-algebra/kernels/mvt/mvt.cpp)
target_link_libraries(mvt PUBLIC Halide::Halide)

# linear algebra - solvers
add_executable(cholesky linear-algebra/solvers/cholesky/cholesky.cpp)
target_link_libraries(cholesky PUBLIC Halide::Halide)
add_executable(durbin linear-algebra/solvers/durbin/durbin.cpp)
target_link_libraries(durbin PUBLIC Halide::Halide)
add_executable(gramschmidt linear-algebra/solvers/gramschmidt/gramschmidt.cpp)
target_link_libraries(gramschmidt PUBLIC Halide::Halide)
add_executable(lu linear-algebra/solvers/lu/lu.cpp)
target_link_libraries(lu PUBLIC Halide::Halide)
add_executable(ludcmp linear-algebra/solvers/ludcmp/ludcmp.cpp)
target_link_libraries(ludcmp PUBLIC Halide::Halide)
add_executable(trisolv linear-algebra/solvers/trisolv/trisolv.cpp)
target_link_libraries(trisolv PUBLIC Halide::Halide)

# medley
add_executable(deriche medley/deriche/deriche.cpp)
target_link_libraries(deriche PUBLIC Halide::Halide)
add_executable(floyd-warshall medley/floyd-warshall/floyd-warshall.cpp)
target_link_libraries(floyd-warshall PUBLIC Halide::Halide)
add_executable(nussinov medley/nussinov/nussinov.cpp)
target_link_libraries(nussinov PUBLIC Halide::Halide)

# # stencils
add_executable(adi stencils/adi/adi.cpp)
target_link_libraries(adi PUBLIC Halide::Halide)
add_executable(fdtd-2d stencils/fdtd-2d/fdtd-2d.cpp)
target_link_libraries(fdtd-2d PUBLIC Halide::Halide)
add_executable(heat-3d stencils/heat-3d/heat-3d.cpp)
target_link_libraries(heat-3d PUBLIC Halide::Halide)
add_executable(jacobi-1d stencils/jacobi-1d/jacobi-1d.cpp)
target_link_libraries(jacobi-1d PUBLIC Halide::Halide)
add_executable(jacobi-2d stencils/jacobi-2d/jacobi-2d.cpp)
target_link_libraries(jacobi-2d PUBLIC Halide::Halide)
add_executable(seidel-2d stencils/seidel-2d/seidel-2d.cpp)
target_link_libraries(seidel-2d PUBLIC Halide::Halide)
