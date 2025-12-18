#!/bin/bash

find results/compilot/ -mindepth 1 -exec ./scripts/adapt-compilot.py {} --output-root=results/optimization/20251215/tiramisu \;

find results/optimization/20251215/tiramisu/ -mindepth 1 -name '*_XLARGE.csv' | while read -r file; do
	sed -i 's/,XLARGE/,EXTRALARGE/g' "$file"
	mv "$file" "${file/_XLARGE.csv/_EXTRALARGE.csv}"
done
