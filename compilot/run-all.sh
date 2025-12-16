#!/bin/bash

#script expects to be run from the container
TIRAMISU_DIR="/tiramisu"
Execution_count=5

REQUESTED_TYPE=${1:-}
TESTED_TYPES=("MINI" "SMALL" "MEDIUM" "LARGE" "XLARGE")

case "$REQUESTED_TYPE" in
([mM][iI][nN][iI])
	TESTED_TYPES=("MINI")
	;;
([sS][mM][aA][lL][lL])
	TESTED_TYPES=("SMALL")
	;;
([mM][eE][dD][iI][uU][mM])
	TESTED_TYPES=("MEDIUM")
	;;
([lL][aA][rR][gG][eE])
	TESTED_TYPES=("LARGE")
	;;
([xX][lL][aA][rR][gG][eE])
	TESTED_TYPES=("XLARGE")
	;;
([aA][lL][lL])
	# keep all types
	;;
(*)
	if [ -z "$REQUESTED_TYPE" ]; then
		echo "Error: TYPE argument not provided."
	else
		echo "Error: Invalid TYPE argument '$REQUESTED_TYPE'."
	fi

	echo "Usage: $0 <TYPE=MINI|SMALL|MEDIUM|LARGE|XLARGE|ALL>"
	exit 1
	;;
esac


export TIRAMISU_ROOT=${TIRAMISU_DIR}
export LD_LIBRARY_PATH=/usr/local/lib:/tiramisu/build:/tiramisu/3rdParty/isl/lib:/tiramisu/3rdParty/llvm/lib${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
export NB_EXEC=${Execution_count}

function run_case {
	(
		local type="$1" # MINI, SMALL, MEDIUM, LARGE, XLARGE
		local name="$2"

		echo "Running case: ${type} ${name}" >&2

		cd ${TIRAMISU_ROOT}/benchmarks/ || exit 1
		entire_out=$("./compile_and_run_benchmarks.sh" "autoscheduler_benchmarks/polybench/${type}/function_${name}_${type}" "function_${name}_${type}")
		times=$(echo "${entire_out}" | tail -2 | head -1)

		echo "${name};${type};${times}"
	)
}

echo "name;type;execution_times_ms"

for tested_type in "${TESTED_TYPES[@]}"; do
	available_functions=$(ls ${TIRAMISU_ROOT}/benchmarks/autoscheduler_benchmarks/polybench/${tested_type}/ | sed -e 's/function_//' -e "s/_${tested_type}//")
	for func in ${available_functions}; do
		run_case "${tested_type}" "${func}"
	done
done
