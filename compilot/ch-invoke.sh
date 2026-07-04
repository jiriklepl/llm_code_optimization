#!/bin/bash

set -euo pipefail

possible_types=(MINI SMALL MEDIUM LARGE XLARGE ALL)

if [[ $# -ne 1 ]] || [[ " ${possible_types[*]} " != *" ${1:-} "* ]]; then
  echo "Error: TYPE argument not correct."
  echo "Usage: $0 <TYPE=MINI|SMALL|MEDIUM|LARGE|XLARGE>"
  exit 1
fi

TYPE=$1

results_base_dir=../results/compilot
timestamp=$(date +"%Y%m%d_%H%M%S")
csv_result_name=$results_base_dir/$timestamp-$TYPE.csv

echo "result will be in $csv_result_name" >&2
mkdir -p "$results_base_dir"

cmd="ch-run imgdir -b ./copied-tiramisu/:/tiramisu -- /tiramisu/benchmarks/run-all.sh $TYPE"

# output the error output of the command to std out and the std out of the command to the result file
$cmd 3>&1 1>"$csv_result_name" 2>&3
