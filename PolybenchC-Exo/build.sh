#!/usr/bin/env bash

set -uo pipefail

BUILD_DIR=${BUILD_DIR:-build}
DATASET_SIZE=${DATASET_SIZE:-LARGE}
DATA_TYPE=${DATA_TYPE:-FLOAT}

MODULE_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
EXO_DIR=${EXO_DIR:-"$ROOT_DIR/submodules/exo"}

CSTANDARD=${CSTANDARD:-c23}
CFLAGS=${CFLAGS:-"-O3 -std=$CSTANDARD -fopenmp -pthread -march=native -Wfatal-errors -mtune=native -D NDEBUG -D POLYBENCH_TIME -D POLYBENCH_DUMP_ARRAYS -D ${DATASET_SIZE}_DATASET -D DATA_TYPE_IS_$DATA_TYPE -D _POSIX_C_SOURCE=200809L ${CFLAGS:-} "}
INCLUDES=${INCLUDES:-"-I$MODULE_DIR/utilities -I$MODULE_DIR/build ${INCLUDES:-} "}
LDFLAGS=${LDFLAGS:-"-lm -lgomp ${LDFLAGS:-} "}

temp_dir=$(mktemp -d)
trap 'echo "Removing temp dir $temp_dir"; rm -rf "$temp_dir"' EXIT

case "$DATA_TYPE" in
  (FLOAT)
	DATA_TYPE=f32 ;;
  (DOUBLE)
	DATA_TYPE=f64 ;;
  (INT)
	DATA_TYPE=i32 ;;
  (LONG)
	DATA_TYPE=i64 ;;
  (*)
	echo "Unsupported DATA_TYPE: $DATA_TYPE" >&2
	echo "Supported types: FLOAT, DOUBLE, INT, LONG" >&2
	exit 1
	;;
esac

if [[ ! -d "$EXO_DIR/src" ]]; then
  echo "Exo submodule not found at $EXO_DIR/src" >&2
  echo "Run: git submodule update --init --depth 1 submodules/exo" >&2
  exit 1
fi

# Install Python dependencies into a local .venv
python3 -m venv "$ROOT_DIR/.venv" && source "$ROOT_DIR/.venv/bin/activate" >/dev/null 2>&1

pip install --upgrade pip >/dev/null 2>&1
pip install -r "$EXO_DIR/requirements.txt" >/dev/null 2>&1
pip install -e "$EXO_DIR" >/dev/null 2>&1

# Get Exo source files
pushd "$MODULE_DIR" >/dev/null || exit 1
mkdir -p "$BUILD_DIR" linear-algebra datamining stencils medley
sources=$(find linear-algebra datamining stencils medley -name '*.c')

if [ "${1:-}" == "clean" ]; then
	echo "Performing a clean build..." >&2
	rm -rf "${BUILD_DIR:?}"/* || true
	shift || true
fi

if [ -n "${1:-}" ]; then
	algorithm=$1
	shift || true
else
	algorithm="all"
fi

# Generate C code from Exo sources
for src in $sources; do
    base_name=$(basename "${src}" .c)
	base_name_c=${base_name//-/_}  # Replace hyphens with underscores

	if [ "$algorithm" != "all" ] && [ "$algorithm" != "$base_name" ]; then
		continue
	fi

    echo "Processing ${base_name}" >&2
    out_dir="$BUILD_DIR/generated/$(basename "$(dirname "${src}")")"

    # header file path
    src_h="${src%.c}.h"

    # resolve header symbolic link
    if [ -L "${src_h}" ]; then
        src_h="$(readlink -f "${src_h}")"
    fi

    mkdir -p "${out_dir}"

    # Skip if "${out_dir}/${base_name}.c" exists and is newer than "${src}" and "${src_h}"
    if ! [[ -f "${out_dir}/${base_name}.c" && "${out_dir}/${base_name}.c" -nt "${src}" && "${out_dir}/${base_name}.c" -nt "${src_h}" ]]; then
		tmp_src="$temp_dir/${base_name}.py"
		sed  -n '/EXO START/,/EXO END/{//!{s/DATA_TYPE/'"$DATA_TYPE"'/g;p}}' "${src}" > "${tmp_src}"

		if [[ ! -s "${tmp_src}" ]]; then
            echo "Error: No Exo code found in ${src}" >&2
            continue
		fi

		python3 - <<PY
from pathlib import Path
import importlib.util
src_path = Path('${tmp_src}').resolve()
spec = importlib.util.spec_from_file_location('${base_name}', src_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if hasattr(module, 'kernel_${base_name_c}'):
	proc = getattr(module, 'kernel_${base_name_c}')
	out = Path('${out_dir}')
	proc.compile_c(out, '${base_name}')
	print(f"Generated C for '${base_name}' in", out.resolve())
else:
	print(f"No proc named 'kernel_${base_name_c}' found in '${src}'")
PY
		if [ "$?" -ne 0 ]; then
			echo "Error: Generating C code for ${src} failed" >&2
			continue
		fi
	else
		echo "Skipping code generation for ${src}, up-to-date." >&2
	fi

	# Compile the generated C code
	if ! [[ -f "$BUILD_DIR/${base_name}" && "$BUILD_DIR/${base_name}" -nt "${src}" && "$BUILD_DIR/${base_name}" -nt "${out_dir}/${base_name}.c" && "$BUILD_DIR/${base_name}" -nt "utilities/polybench.c" ]]; then
		gcc -o "$BUILD_DIR/${base_name}" ${CFLAGS} ${INCLUDES} "${src}" "${out_dir}/${base_name}.c" utilities/polybench.c ${LDFLAGS} -include "math.h" -include "${src_h}" -lm
		if [ "$?" -ne 0 ]; then
			echo "Error: Compiling ${base_name} failed" >&2
			continue
		fi
	else
		echo "Skipping compilation for ${src}, up-to-date." >&2
	fi

    echo "Built ${base_name} executable."
done

popd >/dev/null || exit 1

echo "Exo build complete. Generated sources in: ${MODULE_DIR}/build"
