[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19635706.svg)](https://doi.org/10.5281/zenodo.19635706)

# Artifact: Effect of Abstractions and Prompting Strategies on LLM-Guided High-Performance Optimizations

This repository contains the replication package for the paper "Effect of Abstractions and Prompting Strategies on LLM-Guided High-Performance Optimizations" submitted to HLPP 2026. The package includes all necessary code, scripts, and instructions to reproduce the results presented in the paper.

```bibtex
@inproceedings{ TODO }
```

## 🚀 Getting started

To prepare the replication package, run the following commands:

```bash
git clone ... replication-package
cd replication-package

./configure.sh
```

## 📋 Prerequisites

- GCC 15.2 or higher with OpenMP support.
- Clang/LLVM 22.1.8 with Polly and the matching LLVM OpenMP runtime for the optional Polly baseline.
- CMake 4.12 or higher.
- Python 3.13 or higher.
- Docker or Charliecloud installed on your system.
- An OpenAI API key for accessing the GPT models, if you plan to run the code generation tasks.

The experiments were performed on a dual-socket system equipped with Intel Xeon Gold 6130 processors, each comprising 16 cores with 2-way hyper-threading (64 threads in total). The 20260704 C, Noarr, Halide, and Exo measurements were compiled with GCC 15.2 and GNU OpenMP. The Polly baseline was compiled with Clang/LLVM 22.1.8, Polly passes, and LLVM `libomp`. CMake 4.12 was used for the CMake-based builds.

## 📂 Project Structure

The project is organized as follows:

- [PolybenchC-4.2.1](./PolybenchC-4.2.1): Contains the [Polybench](https://sourceforge.net/projects/polybench/files/) benchmark suite used for evaluation as baseline, and all necessary scripts to build and run them.
- [PolybenchC-Polly](./PolybenchC-Polly): Builds the same PolyBench C sources with Clang/Polly as a compiler baseline.
- [PolybenchC-Exo](./PolybenchC-Exo): Contains the translated Polybench benchmark translated to the [Exo](https://github.com/exo-lang/exo) language and all necessary scripts to build and run them.
- [PolybenchC-Halide](./PolybenchC-Halide): Contains the translated Polybench benchmark translated to the [Halide](https://github.com/halide/Halide) language and all necessary scripts to build and run them.
- [PolybenchC-Noarr](./PolybenchC-Noarr): Contains the translated Polybench benchmark transcribed using the [Noarr](https://github.com/jiriklepl/noarr-structures) framework and all necessary scripts to build and run them.
- [optimization](./optimization): Contains the template project for optimized versions of the above benchmarks.
- [from_model](./from_model): Contains the template project for optimized versions that use the abstract optimization plans.
- [compilot](./compilot): Contains the code and scripts to run the Tiramisu baseline.
- [submodules](./submodules): Contains the git submodules used in the project: Exo, Halide, Noarr.
- [patches](./patches): Contains patches applied to the submodules.
- [prompts](./prompts): Contains the prompt templates used for querying the LLMs.
- [gpt-querying](./gpt-querying): Contains the code and scripts for querying the LLMs. The OpenAI API should be stored at [gpt-querying/.env](./gpt-querying/.env).
- [requests](./requests): Contains the batch request files sent to the LLMs.
- [responses](./responses): Contains the responses received from the LLMs.
- [results](./results): Contains the measurement results and visualizations (note that `results/*/20260704` is the current measurement set with detailed validity results in the included CSV files).
  - [results/translation](./results/translation): Contains the results of the code translation tasks.
    - [results/translation/20251118_155818](./results/translation/20251118_155818) is the special preselection run used to choose the translated baselines. It is intentionally retained and is not superseded by the newer measurement results.
  - [results/optimization](./results/optimization): Contains the results of the optimization tasks.
  - [results/plots](./results/plots): Contains the plots generated from the results.
- [scripts](./scripts): Contains utility scripts for analysis and visualization.

## 🗺️ Paper section to artifact map

This map links the manuscript methodology and evaluation sections to the main
artifact files. It is intended for traceability; the paper cites the replication
package, while low-level paths are documented here.

- Methodology / translation: prompt templates are in `prompts/translation/`;
  request and response batches are in `requests/openai/translation/` and
  `responses/openai/translation/`; parsed generated translations are in
  `generated/{exo,halide,noarr}/full/<benchmark>-<attempt>/code.cpp`;
  measurement outputs are in `results/translation/20260704/`.
- Methodology / direct optimization: prompt templates are in
  `prompts/optimization/`; request and response batches are in
  `requests/openai/optimization/` and `responses/openai/optimization/`; parsed
  generated attempts are in
  `optimization/<framework>_<prompt>/full/<benchmark>-<attempt>/code.cpp` for
  the prompt/framework naming used by the batch files; measurement
  outputs are in `results/optimization/20260704/`.
- Methodology / abstract optimization plan: plan prompts, requests, and
  responses are in `prompts/to_model/`, `requests/openai/to_model/`, and
  `responses/openai/to_model/`; parsed plans are in
  `to_model/{c,exo,halide,noarr}/full/<benchmark>-<attempt>/code.cpp`.
  Plan-application prompts, requests, and responses are in `prompts/from_model/`,
  `requests/openai/from_model/`, and `responses/openai/from_model/`; parsed
  generated implementations are in
  `from_model/{c,exo,halide,noarr}/full/<benchmark>-<attempt>/code.cpp`.
- Methodology / validation and measurement: the collected per-size validation
  and timing CSVs are `validation_*.csv` and `times_*.csv` under
  `results/translation/20260704/` and `results/optimization/20260704/`.
  The scripts that produce these files are `measure_translations.sh`,
  `measure_optimizations.sh`, and the analysis scripts in `scripts/`.
- Evaluation / translation results: the translation-performance and validity
  claims are backed by `results/translation/20260704/` and the corresponding
  plot tables under `results/plots/20260704/`.
- Evaluation / optimization results and RQ answers: aggregate prompt/framework
  summaries are in `results/plots/20260704-extralarge/framework_prompt_summary.csv`,
  invalid-attempt diagnostics are in
  `results/plots/20260704-extralarge/failure_counts.csv`, best-of-`k` budget
  curves are in `results/plots/20260704-extralarge/best_of_k.csv` and
  `results/plots/20260704-extralarge/best_of_k_budget.pdf`, and per-attempt
  speedups are in `results/plots/20260704-extralarge/speedups_datasets-EXTRALARGE.csv`.
- Evaluation / RQ5 qualitative inspection: direct-C All-Hints examples include
  `optimization/c_all_hints/full/doitgen-05/code.cpp`,
  `optimization/c_all_hints/full/3mm-01/code.cpp`, and
  `optimization/c_all_hints/full/correlation-02/code.cpp`; their measured
  EXTRALARGE rows are in
  `results/plots/20260704-extralarge/speedups_datasets-EXTRALARGE.csv`.
  These files are starting points for inspecting generated low-level rewrites,
  not separate aggregate evidence.
- Evaluation / alternative baselines and prior-work context: fixed-baseline
  comparisons are in `results/plots/20260704-extralarge-tiramisu/`,
  `results/plots/20260704-extralarge-polly/`, and
  `results/plots/20260704-extralarge-min-c-polly/`; raw ComPilot/Tiramisu
  measurement exports are in `results/compilot/`.

## 🤖 Generating the LLM solutions

For the initial translation task, run the following set of commands:

```bash
# Prepare the batch request files
. functions && generate

# Prepare the package for OpenAI requests
cp gpt-querying/.env.example gpt-querying/.env
# (get an OpenAI API key and put it in `gpt-querying/.env`)

# Optional: override the default batch generation settings
export GPT_QUERYING_PROVIDER="openai"
export GPT_QUERYING_REPETITIONS="5"
export GPT_QUERYING_MODEL="gpt-5.1-2025-11-13"
export GPT_QUERYING_REASONING_EFFORT="high"
export GPT_QUERYING_VERBOSITY="medium"

# Send requests to OpenAI (can cost over $50 and take almost 2 hours)
./collect_translations.sh

# Measure the translations on the (SMALL, MEDIUM, LARGE) input sizes (can take several hours)
./measure_translations.sh frameworks test

# Choose the translations used for optimization
#   (replace the last argument with the new folder in `./results/translation`)
./choose_translations.sh do TEST_RESULTS
# use `./choose_translations.sh dry TEST_RESULTS 2>/dev/null` to see the effect first
```

Then, to run the (straightforward) optimization tasks with various prompting:

```bash
# Regenerate the batch request files (updates the optimization requests)
. functions && generate

# Send naive requests to OpenAI (can cost over $50 and take almost 2 hours)
./collect_optimizations.sh naive
# Send all_hints requests to OpenAI (can cost over $50 and take almost 2 hours)
./collect_optimizations.sh all_hints
# Send choose_hints requests to OpenAI (can cost over $50 and take almost 2 hours)
./collect_optimizations.sh choose_hints
```

And finally, for the tasks involving the abstract optimization tasks:

```bash
# Send to_model requests to OpenAI (can cost over $50 and take almost 2 hours)
#   (this prepares the abstract optimization plans discussed in the paper)
./collect_models.sh


# Regenerate the batch request files (updates the optimization requests)
. functions && generate

# Send from_model requests to OpenAI (can cost over $50 and take almost 2 hours)
#   (this prepares the abstract optimization plans discussed in the paper)
./collect_optimizations.sh from_model
```

## ⏱️ Measuring the LLM solutions

For measuring the translation task, run the following:

```bash
./measure_translations.sh frameworks all # (can take days)
```

And for the optimizations tasks:

```bash
./measure_optimizations.sh all naive all # (can take days)
./measure_optimizations.sh all all_hints all # (can take days)
./measure_optimizations.sh all choose_hints all # (can take days)
./measure_optimizations.sh all from_model all # (can take days)
```

The measurement scripts use the `DATA_TYPE` environment variable to select the
PolyBench datatype. It defaults to `DOUBLE`, matching the Tiramisu baseline
reported in the reference paper. To run a separate datatype
configuration, set it explicitly for both translation and optimization
measurements:

```bash
DATA_TYPE=FLOAT ./measure_translations.sh frameworks all
DATA_TYPE=FLOAT ./measure_optimizations.sh all all_hints all
```

Measurement CSVs include a `data_type` column, and resume/skip checks also use
that value. This prevents results from different datatype configurations from
being mixed in the same report directory.

Then collect all folders filled by these scripts into two new folders `TRANSLATION_DIR` and `OPTIMIZATION_DIR` (choose any names)

Our current measurement results are in the folders: [results/translation/20260704](./results/translation/20260704) and [results/optimization/20260704](./results/optimization/20260704).

### Compiler configuration used for the reported results

The 20260704 measurements use `DATA_TYPE=DOUBLE`. The standard C, Noarr, and
Halide builds use CMake `Release` builds with dataset and datatype defines,
OpenMP enabled, and native CPU flags (`-fopenmp -pthread -march=native
-mtune=native`). The Exo build generates C from Exo and then compiles with GCC
using `-O3`, OpenMP, native CPU flags, `POLYBENCH_TIME`,
`POLYBENCH_DUMP_ARRAYS`, and matching dataset/datatype defines.

The Polly baseline uses `PolybenchC-Polly/build.sh`, whose default compiler is
`clang-22`. It enables `-mllvm -polly`,
`-mllvm -polly-vectorizer=stripmine`, and `-mllvm -polly-parallel`, and uses
LLVM OpenMP via `-fopenmp=libomp`. Set `POLLY_C_COMPILER` if the Polly-capable
Clang executable has a different name or path.

The Tiramisu baseline is measured inside the Compilot container environment;
the exact container recipe is in [compilot/Dockerfile.compilot](./compilot/Dockerfile.compilot).

### Measuring the Polly baseline

Polly is a fixed compiler baseline, not an LLM-generated translation. It is
therefore built and measured once per dataset size, with repetition `1` and
version `standard`, and its output is validated against the standard C
implementation:

```bash
./measure_translations.sh polly all
```

The default `./measure_translations.sh all all` command includes Polly as well
as the LLM-generated Noarr, Halide, and Exo translations. Set
`POLLY_C_COMPILER` to select another installed Clang executable that provides
the Polly passes; the default is `clang-22`. Polly measurements use the same
`DATA_TYPE` setting as the translation measurements.

### Measuring the Tiramisu baseline

In our paper, we compare our results to prior work that uses the Tiramisu framework as a baseline. To measure this baseline, run the following commands:

```bash
cd compilot

# Prepare the docker image for the Tiramisu framework
bash docker-prepare.sh

# Run the Tiramisu measurements inside the docker container
bash docker-invoke.sh
```

Alternatively, if you use Charliecloud, run:

```bash
cd compilot

# Prepare the Charliecloud image for the Tiramisu framework
bash ch-prepare.sh

# Run the Tiramisu measurements inside the Charliecloud container
bash ch-invoke.sh
```

The [compilot/README.md](./compilot/README.md) file contains full instructions on how to run the Tiramisu baseline using Docker or Charliecloud.

The measurements of the Tiramisu baseline then have to be converted using the following command:

```bash
bash translate_compilot.sh
```

Converted Tiramisu measurements are marked with `DATA_TYPE`, defaulting to
`DOUBLE`. For a non-default datatype, run for example:

```bash
DATA_TYPE=FLOAT bash translate_compilot.sh
```

And the converted results should be placed in `TRANSLATION_DIR` from the previous section.

## 📊 Regenerating the results visualizations

To visualize the results, run the following commands (use the names `TRANSLATION_DIR` and `OPTIMIZATION_DIR` from the previous section)

```bash
. functions
python3 scripts/analyze_results.py TRANSLATION_DIR OPTIMIZATION_DIR \
  --ord=tree  --plot=scatter --format=pdf
```

Use `--data-type=DOUBLE` or `--data-type=FLOAT` to restrict analysis to one
datatype when the input directories contain multiple datatype configurations.

The plots used in the paper are in the folder: [results/plots/](./results/plots/) and can be regenerated with:

```bash
bash make_plots.sh
```

The same command writes CSV backing tables and plots to `results/plots/20260704/`
and equivalent single-dataset artifacts to the
`results/plots/20260704-{mini,small,medium,large,extralarge}/` directories.
It also writes EXTRALARGE fixed-baseline comparisons to
`results/plots/20260704-extralarge-tiramisu/` and
`results/plots/20260704-extralarge-polly/`, plus the composite
`results/plots/20260704-extralarge-min-c-polly/` baseline that uses the faster
of standard C and standard Polly per datatype, benchmark, and dataset size. Set
`RESULTS_TAG` to regenerate the same plot family for a different result set.
Polly and Tiramisu are fixed baselines rather than five-attempt LLM result
families, so `make_plots.sh` excludes them from generated-attempt summaries and
plots alternate-baseline comparisons separately.
All backing CSVs that contain measured or derived benchmark results report
`data_type`. Legacy CSVs without this column are interpreted as `DOUBLE` by the
analysis scripts. Aggregations and best-candidate selection are performed
within each datatype, not across datatypes.

Attempt-level artifacts:

- `framework_prompt_summary.csv` has one row per datatype/framework/prompt tuple. `attempt_count` is the number of framework/prompt/benchmark/repetition attempts; `validation_count` and `validation_rate` count attempts that validate on every selected dataset size; `performance_attempt_count` counts validated attempts that also have speedup data; `median_speedup`, `geometric_mean_speedup`, `speedup_q1`, `speedup_q3`, and `speedup_iqr` summarize only performance-eligible attempts; `speedup_at_5` is the conditional best speedup when all five stored attempts are available; `contributing_benchmark_count` counts benchmarks contributing to the speedup aggregate; `benchmark_count` counts covered benchmarks.
- `failure_counts.csv` has one row per datatype/framework/prompt tuple. `invalid_attempt_count` is the number of attempts that did not validate. The reason columns (`compile_error_count`, `runtime_error_count`, `numerical_mismatch_count`, `timeout_count`, `unclassified_invalid_count`) are diagnostic tags on invalid attempts, not a partition: one invalid attempt can contribute to more than one reason count across dataset sizes, so these columns need not sum to `invalid_attempt_count`.
- `best_of_k.csv` has one row per datatype/framework/prompt/budget `k`. It exactly enumerates all subsets of the five stored attempts. `subsets_per_benchmark` is `choose(5,k)`, `subset_evaluation_count` is that value times the benchmark count, `successful_subset_count` counts benchmark/subset evaluations containing at least one validated candidate, `probability_validated_candidate` is their fraction, `conditional_best_speedup*` summarizes best speedup only among successful subset evaluations, and `expected_speedup_with_c_fallback` uses the standard C implementation when no candidate validates or all validated candidates are slower than C. In C-baseline outputs this fallback is 1.0x; in non-C-baseline outputs it is C performance expressed on the selected baseline's speedup scale. `best_of_k_budget.pdf` visualizes these budget curves.

Other CSV backing tables:

- `speedups*.csv` contains per datatype/framework/version/benchmark/repetition/dataset-size speedups relative to the selected baseline, plus the optimized and baseline runtimes used for each ratio.
- `outcomes*.csv` classifies each datatype/framework/version/benchmark/repetition outcome into the categories used for correctness and win-rate plots.
- `shares*.csv` aggregates those outcome categories as percentages across repetitions, grouped by datatype plus framework, version, or framework/version depending on the filename.
- `heatmap_speedups*.csv` is the matrix used by the speedup heatmap; benchmark columns contain aggregated speedups, `geom_mean` is the geometric mean across benchmark columns, and `wins` counts benchmark wins under the selected threshold.
- `heatmap_correctness*.csv` is the matrix used by the correctness heatmap; benchmark columns contain validation shares and `mean_valid` is the arithmetic mean of those shares across benchmark columns.

An attempt is a datatype/framework/prompt/benchmark/repetition tuple and is
validated only if every selected dataset size validates. Its speedup is the
geometric mean across those sizes. All speedups in these artifacts use the
standard C implementation with the matching datatype as the baseline; invalid
attempts remain in coverage and probability denominators but are excluded from
performance aggregates. Conditional best speedup is aggregated over successful
benchmark/subset evaluations with a geometric mean; `speedup_at_5` is its value
for the full set of five attempts.

Across the analysis scripts, ratios from different benchmarks, dataset sizes, prompts, frameworks, or attempts are combined with geometric means. Repeated runtime samples are averaged using arithmetic means. The translation-selection procedure is intentionally exempt from this convention.

The benchmarks `2mm`, `gemm`, `floyd-warshall`, and `heat-3d` were used while
developing and tuning the prompts. They are therefore excluded by default from
every visualization and its backing aggregate tables. The same set appears as
`TEST_BENCHMARKS` in `gpt-querying/main.py` and in the generated
`requests/openai/{translation,optimization,to_model,from_model}/*_test.jsonl`
request families. Pass `--include-tuning-benchmarks` directly to an analysis
script only for diagnostic outputs that intentionally include the tuning set.

## 🔢 Numbers presented in the paper

To regenerate the numbers presented in the paper, run the following command:

```bash
bash paper_numbers.sh | tee paper_numbers.txt
```

The optimization failure-category summary printed by this script is read from
`results/plots/20260704/failure_counts.csv`; it does not rerun experiments or
regenerate LLM outputs.

## 🔄 Regenerating the OpenAI request batch files

This should be done whenever the files in the [prompts](./prompts) folder are edited. Also, when the [responses/openai/to_model](./responses/openai/to_model) responses change.

```bash
. functions
generate
```

## 📥 Extracting the responses from the batch files

To extract the responses from the batch files received from OpenAI, run the following command:

```bash
. functions
generate --parse
```

This populates the following folders with the extracted [responses](./responses):

- [generated](./generated): Contains the generated translations of the benchmarks.
- [optimizations](./optimizations): Contains the generated optimizations of the benchmarks.
- [to_model](./to_model): Contains the generated abstract optimization plans.
- [from_model](./from_model): Contains the generated optimized versions that use the abstract optimization plans.

## 📄 License

This project is available under the MIT License. See the [LICENSE](./LICENSE) file for more details.
