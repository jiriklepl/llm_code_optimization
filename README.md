# Shifting Automated Code Optimization Paradigms with Large Language Models

This repository contains the replication package for the paper: Shifting Automated Code Optimization Paradigms with Large Language Models, submitted to ISC 2026.

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
- CMake 4.12 or higher.
- Python 3.13 or higher.
- Docker or Charliecloud installed on your system.
- An OpenAI API key for accessing the GPT models, if you plan to run the code generation tasks.

The experiments were performed on a dual-socket system equipped with Intel Xeon Gold 6130 processors, each comprising 16 cores with 2-way hyper-threading (64 threads in total). All benchmarks were compiled with GCC 15.2 and CMake 4.12.

## 📂 Project Structure

The project is organized as follows:

- [PolybenchC-4.2.1](./PolybenchC-4.2.1): Contains the [Polybench](https://sourceforge.net/projects/polybench/files/) benchmark suite used for evaluation as baseline, and all necessary scripts to build and run them.
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
- [results](./results): Contains the measurement results and visualizations.
  - [results/translation](./results/translation): Contains the results of the code translation tasks.
  - [results/optimization](./results/optimization): Contains the results of the optimization tasks.
  - [results/plots](./results/plots): Contains the plots generated from the results.
- [scripts](./scripts): Contains utility scripts for analysis and visualization.

## 🤖 Generating the LLM solutions

For the initial translation task, run the following set of commands:

```bash
# Prepare the batch request files
. functions && generate

# Prepare the package for OpenAI requests
cp gpt-querying/.env.example gpt-querying/.env
# (get an OpenAI API key and put it in `gpt-querying/.env`)

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
./collect_optimizations.sh to_model


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

Then collect all folders filled by these scripts into two new folders `TRANSLATION_DIR` and `OPTIMIZATION_DIR` (choose any names)

Our measurement results are in the folders: [results/translation/20251215](./results/translation/20251215) and [results/optimization/20251215](./results/optimization/20251215).

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
bash charlie-prepare.sh

# Run the Tiramisu measurements inside the Charliecloud container
bash charlie-invoke.sh
```

The [compilot/README.md](./compilot/README.md) file contains full instructions on how to run the Tiramisu baseline using Docker or Charliecloud.

The measurements of the Tiramisu baseline then have to be converted using the following command:

```bash
bash translate_compilot.sh
```

And the converted results should be placed in `TRANSLATION_DIR` from the previous section.

## 📊 Regenerating the results visualizations

To visualize the results, run the following commands (use the names `TRANSLATION_DIR` and `OPTIMIZATION_DIR` from the previous section)

```bash
. functions
python3 scripts/analyze_results.py TRANSLATION_DIR OPTIMIZATION_DIR \
  --ord=tree  --plot=scatter --format=pdf
```

The plots used in the paper are in the folder: [results/plots/](./results/plots/) and can be regenerated with:

```bash
bash make_plots.sh
```

## 🔢 Numbers presented in the paper

To regenerate the numbers presented in the paper, run the following command:

```bash
bash paper_numbers.sh | tee paper_numbers.txt
```

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

This project will be made available under the MIT License after the review process is complete.
