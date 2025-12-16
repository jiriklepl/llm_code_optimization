# Experiments with Large Language Models for Code Generation and Optimization

## Getting started

```bash
git clone ... replication-package
cd replication-package

./configure.sh
```

## Generating the LLM solutions

For the initial translation task, run the following set of commands:

```bash
# Prepare the batch request files
. functions && generate

# Prepare the package for OpenAI requests
cp gpt-querying/.env.example gpt-querying/.env
# (get an OpenAI API key and put it in `gpt-querying/.env`

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

## Measuring the LLM solutions

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

## Regenerating the results visualizations

To visualize the results, run the following commands (use the names `TRANSLATION_DIR` and `OPTIMIZATION_DIR` from the previous section)

```bash
. functions
python3 analyze_results.py TRANSLATION_DIR OPTIMIZATION_DIR \
  --ord=tree  --plot=scatter --format=pdf
```

## Regenerating the OpenAI request batch files

This should be done whenever the files in the [prompts](./prompts) folder are edited. Also, when the [responses/openai/to_model](./responses/openai/to_model) responses change.

```bash
. functions
generate
```

## Extracting the responses from the batch files

```bash
. functions
generate --parse
```
