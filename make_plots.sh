#!/bin/bash

set -euo pipefail

. functions
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --output-dir=results/plots/20251215/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --baseline=tiramisu --output-dir=results/plots/20251215-tiramisu/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --dataset=EXTRALARGE --output-dir=results/plots/20251215-extralarge/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --dataset=EXTRALARGE --baseline=tiramisu --output-dir=results/plots/20251215-extralarge-tiramisu/
