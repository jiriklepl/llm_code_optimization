#!/bin/bash

set -euo pipefail

# Do the original plots from data collected on 2025-12-15
. functions
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --output-dir=results/plots/20251215/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=MINI --output-dir=results/plots/20251215-mini/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=SMALL --output-dir=results/plots/20251215-small/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=MEDIUM --output-dir=results/plots/20251215-medium/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=LARGE --output-dir=results/plots/20251215-large/
python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=EXTRALARGE --output-dir=results/plots/20251215-extralarge/

python3 scripts/analyze_results.py results/translation/20251215/ results/optimization/20251215 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --dataset=EXTRALARGE --baseline=tiramisu --output-dir=results/plots/20251215-extralarge-tiramisu/

# Do the plots from data collected on 2026-04-08
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --output-dir=results/plots/20260408/
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=MINI --output-dir=results/plots/20260408-mini/
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=SMALL --output-dir=results/plots/20260408-small/
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=MEDIUM --output-dir=results/plots/20260408-medium/
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=LARGE --output-dir=results/plots/20260408-large/
python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --no-framework=tiramisu --dataset=EXTRALARGE --output-dir=results/plots/20260408-extralarge/

python3 scripts/analyze_results.py results/translation/20260408/ results/optimization/20260408 --ord=tree  --plot=scatter --format=pdf --warmup=2 --count-wins --hide-colorbar --dataset=EXTRALARGE --baseline=tiramisu --output-dir=results/plots/20260408-extralarge-tiramisu/
