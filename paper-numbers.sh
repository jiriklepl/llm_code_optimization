#!/bin/bash

set -euo pipefail

echo "Exo non-valid translations: $(grep  "^exo" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"
echo "Noarr non-valid translations: $(grep  "^noarr" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"
echo "Halide non-valid translations: $(grep  "^halide" <(cat ./results/translation/20251215/*/vali*.csv) | grep -v ,valid, | cut -d, -f2,6 | sort -u | wc -l)"
