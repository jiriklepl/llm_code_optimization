#!/bin/bash

export DATA_TYPE="${DATA_TYPE:-FLOAT}"
provider="${GPT_QUERYING_PROVIDER:-openai}"

set -euo pipefail

reps="${GPT_QUERYING_REPETITIONS:-${GPT_QUERYING_REPS:-5}}"

script_dir=$(cd "$(dirname "$0")" && pwd)

choose_best_script="$script_dir/scripts/choose_best.py"

if [[ ! -f "$choose_best_script" ]]; then
	echo "Error: choose_best.py not found in $script_dir" >&2
	exit 1
fi

results_dir="$script_dir/results/translation"

# Find the newest results folder (each named by timestamp)
# newest_results=$(find "$results_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

# use the results used during the experiments
newest_results="$results_dir/20251118_155818"

mode="${1:-dry}"

if [[ "$mode" != "dry" && "$mode" != "do" && "$mode" != "check" ]]; then
	echo "Usage: $0 [dry|do|check]" >&2
	exit 1
fi

results_folder="${2:-${newest_results}}"

if [[ ! -d "$results_folder" ]]; then
	echo "Error: Results folder '$results_folder' not found" >&2
	exit 1
fi

. functions >&2
provider="$(resolve_provider "${provider}")"

rm -rf generated
cleanup "$mode" "all" "translation"
generate --rep "${reps}" --parse --provider "${provider}"

python3 "$choose_best_script" "$results_folder" | awk -F, '/# Best/{start=1; next} start{ print $1, $2, $3, $4 }' | while read -r framework algorithm repetition validity_score; do
	if [[ -z "$framework" || -z "$algorithm" || -z "$repetition" || "$framework" == "framework" ]]; then
		continue
	fi

	if [[ $validity_score == "0" ]]; then
		invalidate_codes "$mode" "$framework" "translation" "$algorithm"
		continue
	fi

	replace_codes "$mode" "$framework" "translation" "$repetition" "$algorithm"
done

if [[ "$mode" == "do" ]]; then
	echo "Postprocessing all codes for translation benchmark..." >&2
	postprocess_codes "all" "translation"
fi
