# Script for querying LLMs via `gpt51` or `gemma`

## ⚙️ Setup

1. (Optional): Create the virtual environment: `python3 -m venv .venv` and activate it: `source .venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`

Default provider: `gpt51`

* Create an API key: <https://platform.openai.com/api-keys>
* Rename `.env.example` to `.env` and save the API key there

Optional gemma provider:

* Set `GPT_QUERYING_PROVIDER=gemma`
* Set `GEMMA_API_BASE=http://bw01:8000/v1`
* Optionally override `GEMMA_MODEL`, `GEMMA_TEMPERATURE`, and `GEMMA_API_KEY`
* The same helper commands still work; `gemma` executes request files synchronously instead of using async batch polling

## 📝 Usage

The [./main.py](./main.py) is used for generating the batch request files and parsing the responses.

```help
usage: main.py [-h] [--prompts-folder PROMPTS_FOLDER] [--provider PROVIDER]
               [--model MODEL] [--temperature TEMPERATURE]
               [--requests-folder REQUESTS_FOLDER]
               [--responses-folder RESPONSES_FOLDER]
               [--generated-folder GENERATED_FOLDER]
               [--optimization-folder OPTIMIZATION_FOLDER]
               [--to-model-folder TO_MODEL_FOLDER]
               [--from-model-folder FROM_MODEL_FOLDER]
               [--repetitions REPETITIONS] [--parse]

Tool to prepare and process GPT querying batches.

options:
  -h, --help            show this help message and exit
  --prompts-folder PROMPTS_FOLDER
                        Path to the prompts folder.
  --provider PROVIDER   Provider label used for request and response folders.
  --model MODEL         Model identifier written into generated requests.
                        Defaults depend on the provider.
  --temperature TEMPERATURE
                        Sampling temperature for providers that support it
                        (used by gemma).
  --requests-folder REQUESTS_FOLDER
                        Path to the requests folder. Defaults to ../requests/<provider>.
  --responses-folder RESPONSES_FOLDER
                        Path to the responses folder. Defaults to ../responses/<provider>.
  --generated-folder GENERATED_FOLDER
                        Path to the generated folder.
  --optimization-folder OPTIMIZATION_FOLDER
                        Path to the optimization folder.
  --to-model-folder TO_MODEL_FOLDER
                        Path to the to_model folder.
  --from-model-folder FROM_MODEL_FOLDER
                        Path to the from_model folder.
  --repetitions REPETITIONS
                        Number of repetitions per prompt.
  --parse               Do a parse step (otherwise, do a generate phase)
```

The [./run_batch.py](./run_batch.py) executes provider-specific requests and collects the responses.

```help
usage: run_batch.py [-h] [--provider PROVIDER] [--api-key API_KEY] [--api-base API_BASE] [--beta-header BETA_HEADER] [--request REQUEST] [--receive]
                    [--requests-file REQUESTS_FILE] [--responses-dir RESPONSES_DIR] [--completion-window COMPLETION_WINDOW] [--metadata METADATA]

Submit and retrieve provider-specific query jobs.

options:
  -h, --help            show this help message and exit
  --provider PROVIDER   Provider label used for request execution and
                        default response paths.
  --api-key API_KEY     API key. Overrides provider-specific environment
                        variables when supported.
  --api-base API_BASE   Override the API base URL. Defaults depend on the
                        provider.
  --beta-header BETA_HEADER
                        Optional value for provider-specific beta headers
                        (used by gpt51).
  --request REQUEST     Path to a JSONL file to submit.
  --receive             Receive or poll for responses for the selected
                        provider.
  --requests-file REQUESTS_FILE
                        File tracking outstanding batch IDs. Defaults to gpt-querying/requests.<provider>.jsonl.
  --responses-dir RESPONSES_DIR
                        Directory to store provider outputs. Defaults to ../responses/<provider>.
  --completion-window COMPLETION_WINDOW
                        Completion window requested for new gpt51 batches.
  --metadata METADATA   Optional JSON object to attach as gpt51 batch
                        metadata.
```
