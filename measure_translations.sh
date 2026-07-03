#!/bin/bash

export DATA_TYPE="${DATA_TYPE:-DOUBLE}"
CACHE=${CACHE:-}
provider="${GPT_QUERYING_PROVIDER:-openai}"

set -uo pipefail

reps="${GPT_QUERYING_REPETITIONS:-${GPT_QUERYING_REPS:-5}}"

. functions

ensure_data_type_column() {
	local csv_file="$1"
	if [[ ! -f "${csv_file}" ]]; then
		return
	fi
	local header
	header=$(head -n1 "${csv_file}")
	if [[ ",${header}," == *",data_type,"* ]]; then
		return
	fi
	local tmp_file
	tmp_file=$(mktemp)
	awk -v data_type="${DATA_TYPE}" 'NR == 1 { print $0 ",data_type"; next } { print $0 "," data_type }' "${csv_file}" > "${tmp_file}"
	mv "${tmp_file}" "${csv_file}"
}

algorithms=$(find PolybenchC-4.2.1/datamining PolybenchC-4.2.1/linear-algebra PolybenchC-4.2.1/medley PolybenchC-4.2.1/stencils -type f -name "*.c" | \
	sed -n 's|.*/\([^/]*\)\.c$|\1|p' | sort -u)

RUN_POLLY=false

case "${1:-}" in
(noarr)
	FRAMEWORKS_TO_RUN=("noarr") ;;
(polly)
	FRAMEWORKS_TO_RUN=()
	RUN_POLLY=true ;;
(halide)
	FRAMEWORKS_TO_RUN=("halide") ;;
(exo)
	FRAMEWORKS_TO_RUN=("exo") ;;

(all|"")
	FRAMEWORKS_TO_RUN=("noarr" "halide" "exo")
	RUN_POLLY=true ;;

(frameworks)
	FRAMEWORKS_TO_RUN=("noarr" "halide" "exo") ;;

(*)
	echo "Unsupported framework: ${1:-}" >&2
	FRAMEWORKS_TO_RUN=()
	exit 1 ;;
esac

if [[ ${#FRAMEWORKS_TO_RUN[@]} -gt 0 ]]; then
	cleanup "do"
	generate --rep "${reps}" --parse --provider "${provider}"
fi

abs_tolerance="1e-2"
rel_tolerance="2e-7"

results_dir="results/translation"

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

case "${2:-}" in
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
	echo "Unsupported dataset size: ${2:-}" >&2
	DATASET_SIZEs=()
	exit 1 ;;
esac

runs=5

# kill on ^C
trap 'echo "Interrupted! Exiting..."; exit 1' INT

for DATASET_SIZE in "${DATASET_SIZEs[@]}"; do
	export DATASET_SIZE

	# Prepare the baseline
	build "c" "translation" "clean"

	validation_file="${results_dir}/validation_${DATASET_SIZE}.csv"
	times_file="${results_dir}/times_${DATASET_SIZE}.csv"

	error_file="${results_dir}/output_${DATASET_SIZE}.err"
	output_file="${results_dir}/output_${DATASET_SIZE}.out"

	if ! [[ -f "${validation_file}" ]]; then
		echo "framework,algorithm,time_s,speedup,status,repetition,dataset_size,data_type" > "${validation_file}"
	fi

	if ! [[ -f "${times_file}" ]]; then
		echo "framework,version,algorithm,run,time_s,repetition,dataset_size,data_type" > "${times_file}"
	fi

	ensure_data_type_column "${validation_file}"
	ensure_data_type_column "${times_file}"

	if [ "${3:-}" != "noc" ]; then
		for algorithm in ${algorithms}; do
			actual_runs=${runs}

			if grep -q "^c,standard,${algorithm},1,.*,1,${DATASET_SIZE},${DATA_TYPE}$" "${times_file}"; then
				actual_runs=$(( runs - $(grep -c "^c,standard,${algorithm},.*,.*,1,${DATASET_SIZE},${DATA_TYPE}$" "${times_file}") ))
			fi

			if [[ ${actual_runs} -le 0 ]]; then
				echo "Skipping C standard ${algorithm} for dataset size ${DATASET_SIZE}, data type ${DATA_TYPE} (already done)." | tee -a "${output_file}"
				continue
			fi

			run "${algorithm}" "c" "translation" "${actual_runs}" | awk '{print  $0 ",1,'"${DATASET_SIZE}"','"${DATA_TYPE}"'"}' | tee -a "${times_file}"
		done 2>> "${error_file}" | tee -a "${output_file}"
	fi

	if [[ "${RUN_POLLY}" == true ]]; then
		build "polly" "translation" "clean"

		for algorithm in ${algorithms}; do
			actual_runs=${runs}
			if grep -q "^polly,${algorithm},.*,1,${DATASET_SIZE},${DATA_TYPE}$" "${validation_file}"; then
				validation=$(grep "^polly,${algorithm},.*,1,${DATASET_SIZE},${DATA_TYPE}$" "${validation_file}")
				actual_runs=$(( runs - $(grep -c "^polly,standard,${algorithm},.*,.*,1,${DATASET_SIZE},${DATA_TYPE}$" "${times_file}") ))
			else
				validation=$(check_translation "${algorithm}" "polly" "${abs_tolerance}" "${rel_tolerance}" | awk '{print $0 ",1,'"${DATASET_SIZE}"','"${DATA_TYPE}"'"}' | tee -a "${validation_file}")
				echo "${validation}"
			fi

			if [[ ${actual_runs} -le 0 ]]; then
				echo "Skipping Polly baseline ${algorithm} for dataset size ${DATASET_SIZE}, data type ${DATA_TYPE} (already done)." | tee -a "${output_file}"
				continue
			fi

			status_column=$(echo "${validation}" | awk -F, '{print $5}')
			if [[ ${status_column} == "true" || ${status_column} == "valid" ]]; then
				run "${algorithm}" "polly" "translation" "${actual_runs}" | awk '{print $0 ",1,'"${DATASET_SIZE}"','"${DATA_TYPE}"'"}' | tee -a "${times_file}"
			fi
		done 2>> "${error_file}" | tee -a "${output_file}"
	fi

	for framework in "${FRAMEWORKS_TO_RUN[@]}"; do
		for rep in $(seq "${reps}"); do
			invalidate_codes "do" "${framework}" "translation"
			replace_codes "do" "${framework}" "translation" "${rep}"
			postprocess_codes "${framework}" "translation"
			build "${framework}" "translation" "clean"

			for algorithm in ${algorithms}; do
				actual_runs=${runs}
				if grep -q "^${framework},${algorithm},.*,${rep},${DATASET_SIZE},${DATA_TYPE}$" "${validation_file}"; then
					validation=$(grep "^${framework},${algorithm},.*,${rep},${DATASET_SIZE},${DATA_TYPE}$" "${validation_file}")
					actual_runs=$(( runs - $(grep -c "^${framework},standard,${algorithm},.*,.*,${rep},${DATASET_SIZE},${DATA_TYPE}$" "${times_file}") ))
				else
					validation=$(check_translation "${algorithm}" "${framework}" "${abs_tolerance}" "${rel_tolerance}" | awk '{print $0 ",'"${rep}"','"${DATASET_SIZE}"','"${DATA_TYPE}"'"}' | tee -a "${validation_file}")
					echo "${validation}"
				fi

				if [[ ${actual_runs} -le 0 ]]; then
					echo "Skipping ${framework} translation ${algorithm} repetition ${rep} for dataset size ${DATASET_SIZE}, data type ${DATA_TYPE} (already done)." | tee -a "${output_file}"
					continue
				fi

				status_column=$(echo "${validation}" | awk -F, '{print $5}')
				if [[ ${status_column} == "true" || ${status_column} == "valid" ]]; then
					run "${algorithm}" "${framework}" "translation" "${actual_runs}" | awk '{print  $0 ",'"${rep}"','"${DATASET_SIZE}"','"${DATA_TYPE}"'"}' | tee -a "${times_file}"
				fi
			done
		done 2>> "${error_file}" | tee -a "${output_file}"
	done
done
