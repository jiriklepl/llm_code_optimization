#!/bin/bash

set -euo pipefail

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-LARGE}
export DATA_TYPE=${DATA_TYPE:-FLOAT}
export NPROC=${NPROC:-$(nproc)}
export POLLY_C_COMPILER=${POLLY_C_COMPILER:-clang-21}

if ! command -v "${POLLY_C_COMPILER}" >/dev/null 2>&1; then
	echo "Polly compiler '${POLLY_C_COMPILER}' was not found." >&2
	echo "Install Clang/LLVM 21 with Polly and OpenMP, or set POLLY_C_COMPILER." >&2
	exit 1
fi

POLLY_C_COMPILER=$(command -v "${POLLY_C_COMPILER}")
export POLLY_C_COMPILER

if ! "${POLLY_C_COMPILER}" \
	-mllvm -polly \
	-mllvm -polly-vectorizer=stripmine \
	-mllvm -polly-parallel \
	-x c -c /dev/null -o /dev/null; then
	echo "Compiler '${POLLY_C_COMPILER}' does not provide the required Polly passes." >&2
	exit 1
fi

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

# Configure the build
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="$POLLY_C_COMPILER" \
    -D POLYBENCH_DATASET_SIZE="$DATASET_SIZE" \
    -D POLYBENCH_DATA_TYPE="$DATA_TYPE" \
    -G "Unix Makefiles" \
    ..

# Build the project
cmake --build . --config Release -j"$NPROC" --target "$algorithm" -- -k # Keep going until all targets are built
