#!/usr/bin/env bash

version="${1:-}"
START_PATTERN="${2:-}"

frameworks=("c" "noarr" "halide" "exo")

provider="${GPT_QUERYING_PROVIDER:-openai}"

if [ -z "$version" ]; then
	echo "Usage: $0 <version> [start_pattern]"
	exit 1
fi

sleep_interval=$((60 * 2 / 2))

. ./functions
provider="$(resolve_provider "${provider}")"

paths=()
for framework in "${frameworks[@]}"; do
    current_part=0
    while : ; do
        ((current_part++))
        file_path=$(printf "requests/%s/optimization/%s_%s_full_part%02d.jsonl" "$provider" "$framework" "$version" "$current_part")

        if ! [ -f "$file_path" ]; then
            break
        fi

        paths+=("$file_path")
    done
done

echo "Found ${#paths[@]} files to process:"
options=$(for path in "${paths[@]}"; do
    framework=$(echo "$path" | grep -oE '/optimization/([a-z]+)_' | cut -d'/' -f3 | cut -d'_' -f1)
    version=$(echo "$path" | grep -oE '/optimization/([^/]+)_full' | cut -d'/' -f3 | cut -d'_' -f1 --complement | sed 's/_full$//')
    part=$(echo "$path" | grep -oE 'part[0-9]{2}' | cut -d't' -f2)

    echo "$framework $version $part"
done | sort -k3 -n -k1 -k2 | uniq | tr ' ' ',')


started=false
for option in $options; do
    framework=$(echo "$option" | cut -d',' -f1)
    version=$(echo "$option" | cut -d',' -f2)
    part=$(echo "$option" | cut -d',' -f3)

    if [ -n "$START_PATTERN" ] && ! $started; then
        if [[ "$framework $part" == *"$START_PATTERN"* ]]; then
            started=true
        else
            continue
        fi
    else
        started=true
    fi

	optimize "$framework" "$version" "$provider" "full_part$part"
	sleep "$sleep_interval"

	pending=$(receive "$provider" 2>&1 | grep -oE "([0-9]+|No) pending" | cut -d' ' -f1)
	if ! [ "$pending" = "No" ]; then
        echo "Pending batches: $pending"
    fi
done
