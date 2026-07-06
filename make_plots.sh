#!/bin/bash

set -euo pipefail

. functions

results_tag="${RESULTS_TAG:-20260704}"
translation_dir="results/translation/${results_tag}"
optimization_dir="results/optimization/${results_tag}"
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
		--hide-colorbar --no-framework=tiramisu --no-framework=polly "${dataset_args[@]}" \
		--output-dir="$output_dir"
	python3 scripts/analyze_attempts.py "$translation_dir" "$optimization_dir" \
		--format=pdf --warmup=2 --no-framework=tiramisu --no-framework=polly "${dataset_args[@]}" \
		--output-dir="$output_dir"
}

# Plot data collected on 2026-07-04
generate_c_baseline_plots "$plot_root/${results_tag}/"

for dataset_size in MINI SMALL MEDIUM LARGE EXTRALARGE; do
	output_suffix="${dataset_size,,}"
	generate_c_baseline_plots "$plot_root/${results_tag}-$output_suffix/" "$dataset_size"
done

# Alternate baselines keep the generated-attempt summaries, but score them
# against fixed or composite standard baselines.
for baseline in tiramisu polly min-c-polly; do
	baseline_output_dir="$plot_root/${results_tag}-extralarge-${baseline}"
	python3 scripts/analyze_results.py "$translation_dir" "$optimization_dir" \
		--ord=tree --plot=scatter --format=pdf --warmup=2 --count-wins \
		--hide-colorbar --dataset-size=EXTRALARGE --baseline="$baseline" \
		--output-dir="$baseline_output_dir/"
	python3 scripts/analyze_attempts.py "$translation_dir" "$optimization_dir" \
		--format=pdf --warmup=2 --dataset-size=EXTRALARGE \
		--no-framework=tiramisu --no-framework=polly --baseline="$baseline" \
		--output-dir="$baseline_output_dir"
done
