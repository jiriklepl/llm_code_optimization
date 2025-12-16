#!/bin/bash

set -euo pipefail

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-LARGE}
export DATA_TYPE=${DATA_TYPE:-FLOAT}
export NPROC=${NPROC:-$(nproc)}


if [ ! -d "llvm-project" ]; then
	git clone --depth 1 https://github.com/llvm/llvm-project.git
fi

LLVM_PROJECT_ROOT=$(realpath llvm-project)

(
	cd "$LLVM_PROJECT_ROOT"
	cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang;polly;openmp" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
	cmake --build build --config Release -j"$NPROC"
)

# Create the build directory
cmake -E make_directory "$BUILD_DIR"
[ -f "$BUILD_DIR/.gitignore" ] || echo "*" > "$BUILD_DIR/.gitignore"
cd "$BUILD_DIR"

if [ "${1:-}" == "clean" ]; then
	echo "Performing a clean build..." >&2
	rm -rf ./* || true
	shift || true
fi

if [ -n "${1:-}" ]; then
	algorithm=$1
	shift || true
else
	algorithm="all"
fi

omp_lib_path="$LLVM_PROJECT_ROOT/build/lib/"
omp_include_path="$LLVM_PROJECT_ROOT/build/projects/openmp/runtime/src"

# Configure the build
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="$LLVM_PROJECT_ROOT/build/bin/clang" \
    -D CMAKE_C_FLAGS="-D POLYBENCH_TIME -D POLYBENCH_DUMP_ARRAYS -D ${DATASET_SIZE}_DATASET -D DATA_TYPE_IS_$DATA_TYPE -D _POSIX_C_SOURCE=200809L -O3 -DNDEBUG -march=native -mllvm -polly -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel -fopenmp -I $omp_include_path -L $omp_lib_path" \
    -G "Unix Makefiles" \
    ..

# Build the project
cmake --build . --config Release -j"$NPROC" --target "$algorithm" -- -k # Keep going until all targets are built
