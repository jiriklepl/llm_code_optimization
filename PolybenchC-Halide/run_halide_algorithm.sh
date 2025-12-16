#!/bin/bash

prefix="$1"
file="$2"

NUM_RUNS=${NUM_RUNS:-10}

printf "\t%s: " $prefix
"$file" 2>&1 1>/dev/null

for _ in $(seq "$NUM_RUNS"); do
    printf "\t%s: " $prefix
    "$file" 2>&1 1>/dev/null
done
