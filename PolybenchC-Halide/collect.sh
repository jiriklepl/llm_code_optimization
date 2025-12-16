#!/bin/bash

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-EXTRALARGE}
export DATA_TYPE=${DATA_TYPE:-FLOAT}
export USE_SLURM=${USE_SLURM:-0}
export DATA_DIR=${DATA_DIR:-data}

# SLURM settings (if used)
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-kdss}
export SLURM_PARTITION=${SLURM_PARTITION:-mpi-homo-short}
export SLURM_WORKER=${SLURM_WORKER:-w201}
export SLURM_TIMEOUT=${SLURM_TIMEOUT:-"2:00:00"}

POLYBENCH_C_DIR="../PolybenchC-4.2.1"

run_script() {
    if [ "$USE_SLURM" -eq 1 ]; then
        srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" --exclusive -w"$SLURM_WORKER" -t "$SLURM_TIMEOUT" --mem=0 "$@"
    else
        "$@"
    fi
}

( cd "$POLYBENCH_C_DIR" && run_script ./build.sh ) || exit 1
( cd . && run_script ./build.sh ) || exit 1

mkdir -p "$DATA_DIR"

find "$BUILD_DIR" -maxdepth 1 -executable -type f |
while read -r file; do
    filename=$(basename "$file")

    echo "collecting $filename"
    ( run_script ./run_halide_algorithm.sh "Halide" "$BUILD_DIR/$filename" & wait ) > "$DATA_DIR/$filename.log"
	echo "" >> "$DATA_DIR/$filename.log"
    ( run_script ./run_c_algorithm.sh "Baseline" "$POLYBENCH_C_DIR/$BUILD_DIR/$filename" & wait ) >> "$DATA_DIR/$filename.log"
    echo "done"
done
