#!/bin/bash

. functions &> /dev/null

set -euo pipefail

echo "Exo non-valid translations: $(grep  "^exo" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"
echo "Noarr non-valid translations: $(grep  "^noarr" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"
echo "Halide non-valid translations: $(grep  "^halide" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"

generate --parse &> /dev/null

read tot_tokens num_files < <(find optimization/c_all_hints -name "costs.json" -exec jq '.output_tokens, .reasoning_tokens' {} + | grep -E '[0-9]+' | awk '{s+=$1; c++} END {print s " " c}')

echo "Total output tokens (including reasoning) in C all-hints optimization: $tot_tokens"
echo "Number of C all-hints optimization requests: $((num_files / 2))"
echo "Average output tokens per algorithm: $((tot_tokens / 30))"

read tot_tokens num_files < <(find optimization/c_choose_hints -name "costs.json" -exec jq '.output_tokens, .reasoning_tokens' {} + | grep -E '[0-9]+' | awk '{s+=$1; c++} END {print s " " c}')

echo "Total output tokens (including reasoning) in C choose-hints optimization: $tot_tokens"
echo "Number of C choose-hints optimization requests: $((num_files / 2))"
echo "Average output tokens per algorithm: $((tot_tokens / 30))"

printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" "Algorithm" "Reasoning" "Min_Reasoning" "Max_Reasoning" "Output" "Total"
find optimization/c_all_hints -name "costs.json" | while read -r file; do
	name_number=$(basename "$(dirname "$file")")
	# remove the "-[0-9]+$" suffix (the algorithm name may contain further hyphens)
	algorithm_name=$(echo "$name_number" | sed -E 's/-[0-9]+$//')
	reasoning_tokens=$(jq '.reasoning_tokens' "$file")
	output_tokens=$(jq '.output_tokens' "$file")
	total_tokens=$((reasoning_tokens + output_tokens))
	echo "$algorithm_name,$reasoning_tokens,$output_tokens,$total_tokens"
done |
awk -F, '
{
	algorithm=$1
	reasoning=$2
	output=$3
	total=$4

	reasonings[algorithm]+=reasoning
	if (!(algorithm in min_reasonings)) {
		min_reasonings[algorithm]=reasoning
	} else {
		min_reasonings[algorithm]=reasoning < min_reasonings[algorithm] ? reasoning : min_reasonings[algorithm]
	}
	max_reasonings[algorithm]=reasoning > max_reasonings[algorithm] ? reasoning : max_reasonings[algorithm]
	outputs[algorithm]+=output
	totals[algorithm]+=total
}
END {
	for (alg in totals) {
		printf "%-20s %-10d %-10d %-10d %-10d %-10d\n", alg, reasonings[alg], min_reasonings[alg], max_reasonings[alg], outputs[alg], totals[alg]
	}
}' |
sort -t' ' -k6 -n -r
