#!/bin/bash

export DATA_TYPE="${DATA_TYPE:-FLOAT}"
CACHE=${CACHE:-}

set -uo pipefail

reps=5

. functions


cleanup "do"
generate --rep "${reps}" --parse

algorithms=$(find PolybenchC-4.2.1/datamining PolybenchC-4.2.1/linear-algebra PolybenchC-4.2.1/medley PolybenchC-4.2.1/stencils -type f -name "*.c" | \
	sed -n 's|.*/\([^/]*\)\.c$|\1|p' | sort -u)

case "${1:-}" in
(c)
	FRAMEWORKS_TO_RUN=("c") ;;
(noarr)
	FRAMEWORKS_TO_RUN=("noarr") ;;
(halide)
	FRAMEWORKS_TO_RUN=("halide") ;;
(exo)
	FRAMEWORKS_TO_RUN=("exo") ;;

(all|"")
	FRAMEWORKS_TO_RUN=("c" "noarr" "halide" "exo") ;;

(*)
	echo "Unsupported framework: ${1:-}" >&2
	FRAMEWORKS_TO_RUN=()
	exit 1 ;;
esac

case "${2:-}" in
(naive)
	VERSIONS_TO_RUN=("naive") ;;

(arithmetic)
	VERSIONS_TO_RUN=("arithmetic") ;;
(cache)
	VERSIONS_TO_RUN=("cache") ;;
(structure)
	VERSIONS_TO_RUN=("structure") ;;
(parallelism)
	VERSIONS_TO_RUN=("parallelism") ;;

(all_hints)
	VERSIONS_TO_RUN=("all_hints") ;;
(choose_hints)
	VERSIONS_TO_RUN=("choose_hints") ;;

(to_model)
	VERSIONS_TO_RUN=("to_model") ;;
(from_model)
	VERSIONS_TO_RUN=("from_model") ;;

(all|"")
	VERSIONS_TO_RUN=(
		"naive"

		"arithmetic"
		"cache"
		"structure"
		"parallelism"

		"all_hints"
		"choose_hints"

		"to_model"
		"from_model"
	) ;;

(*)
	echo "Unsupported version: ${2:-}" >&2
	VERSIONS_TO_RUN=()
	exit 1 ;;
esac

abs_tolerance="1e-2"
rel_tolerance="2e-7"

results_dir="results/optimization"

if [[ -n "${CACHE}" ]]; then
	if [ -f "${CACHE}" ]; then
		timestamp=$(head -n1 "${CACHE}")
	fi
	if [ -z "${timestamp:-}" ]; then
		timestamp=$(date +"%Y%m%d_%H%M%S")
		echo "${timestamp}" > "${CACHE}"
	fi
else
	timestamp=$(date +"%Y%m%d_%H%M%S")
fi

results_dir="${results_dir}/${timestamp}"
mkdir -p "${results_dir}"

case "${3:-}" in
(mini|MINI)
	DATASET_SIZEs=("MINI") ;;
(small|SMALL)
	DATASET_SIZEs=("SMALL") ;;
(medium|MEDIUM)
	DATASET_SIZEs=("MEDIUM") ;;
(large|LARGE)
	DATASET_SIZEs=("LARGE") ;;
(extralarge|extra|xlarge|largest|EXTRALARGE)
	DATASET_SIZEs=("EXTRALARGE") ;;
(test|TEST)
	DATASET_SIZEs=("SMALL" "MEDIUM" "LARGE") ;;
(notextra|NOTEXTRA)
	DATASET_SIZEs=("MINI" "SMALL" "MEDIUM" "LARGE") ;;
(all|""|ALL)
	DATASET_SIZEs=("MINI" "SMALL" "MEDIUM" "LARGE" "EXTRALARGE") ;;
(*)
	echo "Unsupported dataset size: ${3:-}" >&2
	DATASET_SIZEs=()
	exit 1 ;;
esac

runs=5

# kill on ^C
trap 'echo "Interrupted! Exiting..."; exit 1' INT

for DATASET_SIZE in "${DATASET_SIZEs[@]}"; do
	export DATASET_SIZE

	# Prepare the baseline
	build "c" "standard" "clean"

	validation_file="${results_dir}/validation_${DATASET_SIZE}.csv"
	times_file="${results_dir}/times_${DATASET_SIZE}.csv"

	error_file="${results_dir}/output_${DATASET_SIZE}.err"
	output_file="${results_dir}/output_${DATASET_SIZE}.out"

	if ! [[ -f "${validation_file}" ]]; then
		echo "framework,version,algorithm,time_s,speedup,valid,repetition,dataset_size" > "${validation_file}"
	fi

	if ! [[ -f "${times_file}" ]]; then
		echo "framework,version,algorithm,run,time_s,repetition,dataset_size" > "${times_file}"
	fi

	for framework in "${FRAMEWORKS_TO_RUN[@]}"; do
		for rep in $(seq "${reps}"); do
			for version in "${VERSIONS_TO_RUN[@]}"; do
				# Prepare the optimized version
				invalidate_codes "do" "${framework}" "${version}"
				replace_codes "do" "${framework}" "${version}" "${rep}"
				postprocess_codes "${framework}" "${version}"
				build "${framework}" "${version}" "clean"

				for algorithm in ${algorithms}; do
					actual_runs=${runs}
					if grep -q "^${framework},${version},${algorithm},.*,${rep},${DATASET_SIZE}$" "${validation_file}"; then
						validation=$(grep "^${framework},${version},${algorithm},.*,${rep},${DATASET_SIZE}$" "${validation_file}")
						actual_runs=$(( runs - $(grep -c "^${framework},${version},${algorithm},.*,${rep},${DATASET_SIZE}$" "${times_file}") ))
					else
						validation=$(check_optimization "${algorithm}" "${framework}" "${version}" "${abs_tolerance}" "${rel_tolerance}" | awk '{print $0 ",'"${rep}"','"${DATASET_SIZE}"'"}' | tee -a "${validation_file}")
						echo "${validation}"
					fi

					if [[ ${actual_runs} -le 0 ]]; then
						echo "Skipping ${framework} ${version} ${algorithm} repetition ${rep} for dataset size ${DATASET_SIZE} (already done)." | tee -a "${output_file}"
						continue
					fi

					valid_column=$(echo "${validation}" | awk -F, '{print $6}')
					if [[ ${valid_column} == "true" || ${valid_column} == "valid" ]]; then
						run "${algorithm}" "${framework}" "${version}" "${runs}" | awk '{print  $0 ",'"${rep}"','"${DATASET_SIZE}"'"}' | tee -a "${times_file}"
					fi
				done
			done 2>> "${error_file}" | tee -a "${output_file}"
		done
	done
done
