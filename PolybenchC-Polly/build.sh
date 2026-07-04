#!/bin/bash

set -euo pipefail

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-LARGE}
export DATA_TYPE=${DATA_TYPE:-DOUBLE}
export NPROC=${NPROC:-$(nproc)}
export POLLY_C_COMPILER=${POLLY_C_COMPILER:-clang-21}
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! command -v "${POLLY_C_COMPILER}" >/dev/null 2>&1; then
	echo "Polly compiler '${POLLY_C_COMPILER}' was not found." >&2
	echo "Install Clang/LLVM 21 with Polly and OpenMP, or set POLLY_C_COMPILER." >&2
	exit 1
fi

POLLY_C_COMPILER=$(command -v "${POLLY_C_COMPILER}")
export POLLY_C_COMPILER

OPENMP_CMAKE_ARGS=()
POLLY_OPENMP_FLAGS="-fopenmp=libomp"
POLLY_OPENMP_LIBRARY=$("${POLLY_C_COMPILER}" -print-file-name=libomp.so 2>/dev/null || true)

if [ -z "${POLLY_OPENMP_LIBRARY}" ] || [ "${POLLY_OPENMP_LIBRARY}" = "libomp.so" ] || [ ! -f "${POLLY_OPENMP_LIBRARY}" ]; then
	for candidate in \
		"$(dirname "${POLLY_C_COMPILER}")/../lib/libomp.so" \
		"$(dirname "${POLLY_C_COMPILER}")/../lib/llvm/libomp.so" \
		"/usr/lib/llvm-21/lib/libomp.so" \
		"/usr/lib/x86_64-linux-gnu/libomp.so" \
		"/lib/x86_64-linux-gnu/libomp.so"; do
		if [ -f "${candidate}" ]; then
			POLLY_OPENMP_LIBRARY=$(realpath "${candidate}")
			break
		fi
	done
fi

if [ -f "${POLLY_OPENMP_LIBRARY}" ]; then
	POLLY_OPENMP_LIBRARY_DIR=$(dirname "${POLLY_OPENMP_LIBRARY}")
	OPENMP_CMAKE_ARGS+=(
		"-DOpenMP_C_FLAGS:STRING=${POLLY_OPENMP_FLAGS}"
		"-DOpenMP_C_LIB_NAMES:STRING=omp"
		"-DOpenMP_omp_LIBRARY:FILEPATH=${POLLY_OPENMP_LIBRARY}"
		"-DCMAKE_BUILD_RPATH:STRING=${POLLY_OPENMP_LIBRARY_DIR}"
		"-DCMAKE_INSTALL_RPATH:STRING=${POLLY_OPENMP_LIBRARY_DIR}"
	)

	POLLY_RESOURCE_DIR=$("${POLLY_C_COMPILER}" --print-resource-dir 2>/dev/null || true)
	if [ -f "${POLLY_RESOURCE_DIR}/include/omp.h" ]; then
		OPENMP_CMAKE_ARGS+=("-DOpenMP_C_INCLUDE_DIR:PATH=${POLLY_RESOURCE_DIR}/include")
	fi
else
	echo "Could not find libomp for '${POLLY_C_COMPILER}'." >&2
	echo "Install the matching LLVM OpenMP runtime, e.g. libomp-21-dev, or expose libomp.so to the compiler." >&2
	exit 1
fi

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
    "${OPENMP_CMAKE_ARGS[@]}" \
    -G "Unix Makefiles" \
    "$SOURCE_DIR"

# Build the project
cmake --build . --config Release -j"$NPROC" --target "$algorithm" -- -k # Keep going until all targets are built
