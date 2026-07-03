#!/bin/bash

set -euo pipefail

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-LARGE}
export DATA_TYPE=${DATA_TYPE:-DOUBLE}
export NPROC=${NPROC:-$(nproc)}

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
    -D CMAKE_C_FLAGS="-D POLYBENCH_TIME -D POLYBENCH_DUMP_ARRAYS -D ${DATASET_SIZE}_DATASET -D DATA_TYPE_IS_$DATA_TYPE -D _POSIX_C_SOURCE=200809L " \
    -G "Unix Makefiles" \
    ..

# Build the project
cmake --build . --config Release -j"$NPROC" --target "$algorithm" -- -k # Keep going until all targets are built
