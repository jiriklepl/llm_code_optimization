#!/bin/bash

set -euo pipefail

# Create the build directory
cmake -E make_directory build
cd build

export DATASET_SIZE=${DATASET_SIZE:-LARGE}
export DATA_TYPE=${DATA_TYPE:-DOUBLE}
export NPROC=${NPROC:-$(nproc)}

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
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:- -D${DATASET_SIZE}_DATASET -DDATA_TYPE_IS_$DATA_TYPE}" \
    -G "Unix Makefiles" \
    ..

# Build the project
cmake --build . --config Release -j"$NPROC" --target "$algorithm" -- -k # Keep going until all targets are built
