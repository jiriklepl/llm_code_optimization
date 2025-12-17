#!/bin/bash

set -euo pipefail

. functions
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --output-dir=results/plots/20251215/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --baseline=tiramisu --output-dir=results/plots/20251215-tiramisu/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --dataset=EXTRALARGE --output-dir=results/plots/20251215-extralarge/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --dataset=EXTRALARGE --baseline=tiramisu --output-dir=results/plots/20251215-extralarge-tiramisu/
