#!/usr/bin/env bash

START_PATTERN="${1:-}"

frameworks=("noarr" "halide" "exo")

provider="${GPT_QUERYING_PROVIDER:-openai}"

sleep_interval=$((60 * 2 / 2))

. ./functions
provider="$(resolve_provider "${provider}")"

paths=()
for framework in "${frameworks[@]}"; do
    current_part=0
    while : ; do
        ((current_part++))
        file_path=$(printf "requests/%s/translation/%s_full_part%02d.jsonl" "$provider" "$framework" "$current_part")

        if ! [ -f "$file_path" ]; then
            break
        fi

        paths+=("$file_path")
    done
done

echo "Found ${#paths[@]} files to process:"
options=$(for path in "${paths[@]}"; do
    framework=$(echo "$path" | grep -oE '/translation/([a-z]+)_' | cut -d'/' -f3 | cut -d'_' -f1)
    part=$(echo "$path" | grep -oE 'part[0-9]{2}' | cut -d't' -f2)

    echo "$framework $part"
done | sort -k2 -n -k1 | uniq | tr ' ' ',')

started=false
for option in $options; do
    framework=$(echo "$option" | cut -d',' -f1)
    part=$(echo "$option" | cut -d',' -f2)

    if [ -n "$START_PATTERN" ] && ! $started; then
        if [[ "$framework $part" == *"$START_PATTERN"* ]]; then
            started=true
        else
            continue
        fi
    else
        started=true
    fi

    translate "$framework" "$provider" "full_part$part"
	sleep "$sleep_interval"

	pending=$(receive "$provider" 2>&1 | grep -oE "([0-9]+|No) pending" | cut -d' ' -f1)
	if ! [ "$pending" = "No" ]; then
	    echo "Pending translations: $pending"
	fi
done
