# Script for querying LLMs via OpenAI API

## ⚙️ Setup

1. (Optional): Create the virtual environment: `python3 -m venv .venv` and activate it: `source .venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`

To use OpenAI (paid API):

* Create an API key: <https://platform.openai.com/api-keys>
* Rename `.env.example` to `.env` and save the API key there

## 📝 Usage

The [./main.py](./main.py) is used for generating the batch request files and parsing the responses.

```help
usage: main.py [-h] [--prompts-folder PROMPTS_FOLDER] [--requests-folder REQUESTS_FOLDER] [--responses-folder RESPONSES_FOLDER] [--generated-folder GENERATED_FOLDER] [--optimization-folder OPTIMIZATION_FOLDER] [--to-model-folder TO_MODEL_FOLDER] [--from-model-folder FROM_MODEL_FOLDER]
               [--repetitions REPETITIONS] [--parse]

Tool to prepare and process GPT querying batches.

options:
  -h, --help            show this help message and exit
  --prompts-folder PROMPTS_FOLDER
                        Path to the prompts folder.
  --requests-folder REQUESTS_FOLDER
                        Path to the requests folder.
  --responses-folder RESPONSES_FOLDER
                        Path to the responses folder.
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

The [./run_batch.py](./run_batch.py) is used for sending the batch requests to OpenAI and collecting the responses.

```help
usage: run_batch.py [-h] [--api-key API_KEY] [--api-base API_BASE] [--beta-header BETA_HEADER] [--request REQUEST] [--receive] [--requests-file REQUESTS_FILE] [--responses-dir RESPONSES_DIR] [--completion-window COMPLETION_WINDOW] [--metadata METADATA]

Submit and retrieve OpenAI Batch API jobs.

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     OpenAI API key. Overrides OPENAI_API_KEY environment variable.
  --api-base API_BASE   Override the API base URL (default: https://api.openai.com/v1).
  --beta-header BETA_HEADER
                        Optional value for the OpenAI-Beta header.
  --request REQUEST     Path to a JSONL file to submit to the Batch API.
  --receive             Attempt to download completed batches listed in the requests file.
  --requests-file REQUESTS_FILE
                        File tracking outstanding batch IDs.
  --responses-dir RESPONSES_DIR
                        Directory to store downloaded batch outputs.
  --completion-window COMPLETION_WINDOW
                        Completion window requested for new batches.
  --metadata METADATA   Optional JSON object to attach as batch metadata.
```
