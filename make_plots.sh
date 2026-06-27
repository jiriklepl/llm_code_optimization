#!/bin/bash

set -euo pipefail

. functions

translation_dir="results/translation/20260408/"
optimization_dir="results/optimization/20260408"
plot_root="results/plots"

generate_c_baseline_plots() {
	local output_dir="$1"
	local dataset_size="${2:-}"
	local dataset_args=()

	if [[ -n "$dataset_size" ]]; then
		dataset_args=(--dataset-size="$dataset_size")
	fi

	python3 scripts/analyze_results.py "$translation_dir" "$optimization_dir" \
		--ord=tree --plot=scatter --format=pdf --warmup=2 --count-wins \
		--hide-colorbar --no-framework=tiramisu "${dataset_args[@]}" \
		--output-dir="$output_dir"
	python3 scripts/analyze_attempts.py "$translation_dir" "$optimization_dir" \
		--format=pdf --warmup=2 --no-framework=tiramisu "${dataset_args[@]}" \
		--output-dir="$output_dir"
}

# Plot data collected on 2026-04-08
generate_c_baseline_plots "$plot_root/20260408/"

for dataset_size in MINI SMALL MEDIUM LARGE EXTRALARGE; do
	output_suffix="${dataset_size,,}"
	generate_c_baseline_plots "$plot_root/20260408-$output_suffix/" "$dataset_size"
done

# Tiramisu has one stored repetition, so an exact best-of-k analysis is not defined.
python3 scripts/analyze_results.py "$translation_dir" "$optimization_dir" \
	--ord=tree --plot=scatter --format=pdf --warmup=2 --count-wins \
	--hide-colorbar --dataset-size=EXTRALARGE --baseline=tiramisu \
	--output-dir="$plot_root/20260408-extralarge-tiramisu/"
