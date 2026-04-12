#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import APIError, OpenAI

logger = logging.getLogger("run_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CURRENT_DIR = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROVIDER = os.getenv("GPT_QUERYING_PROVIDER", "gpt51")
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_API_BASE = "https://api.openai.com/v1"


def read_jsonl_line(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                return json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{filepath} does not contain valid JSONL content.") from exc
    raise ValueError(f"{filepath} appears to be empty.")


def load_requests(filepath: Path) -> list[dict[str, Any]]:
    if not filepath.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in filepath.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid line in %s: %s", filepath, line)
    return records


def save_requests(filepath: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, sort_keys=True) for record in records]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def provider_env_prefix(provider: str) -> str:
    return provider.upper().replace("-", "_")


def default_requests_file(provider: str) -> Path:
    return SCRIPT_DIR / f"requests.{provider}.jsonl"


def default_responses_dir(provider: str) -> Path:
    responses_dir = SCRIPT_DIR.parent / "responses" / provider
    try:
        return Path("..") / responses_dir.relative_to(CURRENT_DIR.parent)
    except ValueError:
        return responses_dir


def get_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    value = getattr(obj, key, None)
    if value is not None:
        return value
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if key in dumped:
            return dumped[key]
    raise KeyError(f"{key} not present in object of type {type(obj)!r}")


def maybe_get_attr(obj: Any, key: str) -> Any | None:
    try:
        return get_attr(obj, key)
    except KeyError:
        return None


@dataclass
class BatchRecord:
    batch_id: str
    input_path: str
    endpoint: str
    completion_window: str
    submitted_at: str
    output_path: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_id": self.batch_id,
            "input_path": self.input_path,
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
            "submitted_at": self.submitted_at,
            "output_path": self.output_path,
        }
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


class CompatibleBatchClient:
    def __init__(self, api_key: str, api_base: str | None = DEFAULT_API_BASE, beta_header: str | None = None) -> None:
        default_headers: dict[str, str] | None = None
        if beta_header:
            default_headers = {"OpenAI-Beta": beta_header}

        init_kwargs: dict[str, Any] = {"api_key": api_key}
        if api_base:
            init_kwargs["base_url"] = api_base
        if default_headers:
            init_kwargs["default_headers"] = default_headers

        self._client = OpenAI(**init_kwargs)

    def upload_batch_file(self, filepath: Path):
        with filepath.open("rb") as file_handle:
            return self._client.files.create(file=file_handle, purpose="batch")

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, Any] | None = None,
    ):
        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata

        return self._client.batches.create(**payload)

    def retrieve_batch(self, batch_id: str):
        return self._client.batches.retrieve(batch_id=batch_id)

    def download_file(self, file_id: str) -> bytes:
        file_content = self._client.files.content(file_id=file_id)
        if hasattr(file_content, "read"):
            data = file_content.read()
            if hasattr(file_content, "close"):
                file_content.close()
            return data
        if hasattr(file_content, "content"):
            return file_content.content
        if isinstance(file_content, bytes):
            return file_content
        if isinstance(file_content, str):
            return file_content.encode("utf-8")

        raise TypeError(f"Unsupported file content type: {type(file_content)!r}")


def submit_request(
    client: CompatibleBatchClient,
    jsonl_path: Path,
    requests_file: Path,
    responses_dir: Path,
    completion_window: str,
    metadata: dict[str, Any] | None,
    provider: str,
) -> BatchRecord:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Request file not found: {jsonl_path}")

    first_line = read_jsonl_line(jsonl_path)
    endpoint = first_line.get("url")
    if not endpoint:
        raise ValueError(f"Request file {jsonl_path} does not specify an endpoint via the 'url' field.")

    jsonl_path = jsonl_path.resolve()

    parent_name = jsonl_path.parent.name
    if jsonl_path.parent.parent.name == provider and jsonl_path.parent.parent.parent.name == "requests":
        responses_dir = responses_dir / parent_name

    logger.info("Uploading %s to %s Files API.", jsonl_path, provider)
    file_response = client.upload_batch_file(jsonl_path)
    file_id = get_attr(file_response, "id")
    logger.info("Uploaded file. file_id=%s", file_id)

    logger.info("Creating batch for endpoint %s.", endpoint)
    batch_response = client.create_batch(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
    )
    batch_id = get_attr(batch_response, "id")
    try:
        status = get_attr(batch_response, "status")
    except KeyError:
        status = None
    logger.info("Created batch %s with status %s.", batch_id, status)

    responses_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{jsonl_path.stem}.jsonl"
    output_path = responses_dir / output_filename

    record = BatchRecord(
        batch_id=batch_id,
        input_path=str(jsonl_path),
        endpoint=endpoint,
        completion_window=completion_window,
        submitted_at=iso_timestamp(),
        output_path=str(output_path),
        metadata=metadata,
    )

    existing = load_requests(requests_file)
    existing.append(record.to_dict())
    save_requests(requests_file, existing)

    logger.info("Stored batch %s in %s.", batch_id, requests_file)
    return record


def receive_batches(
    client: CompatibleBatchClient,
    requests_file: Path,
    responses_dir: Path,
) -> None:
    records = load_requests(requests_file)
    if not records:
        logger.info("No pending batch requests in %s.", requests_file)
        return

    responses_dir.mkdir(parents=True, exist_ok=True)

    remaining: list[dict[str, Any]] = []
    for record in records:
        batch_id = record["batch_id"]
        try:
            batch_info = client.retrieve_batch(batch_id)
        except APIError as exc:
            logger.error("Failed to retrieve batch %s: %s", batch_id, exc)
            record["last_error"] = str(exc)
            remaining.append(record)
            continue
        except Exception as exc:
            logger.error("Unexpected error retrieving batch %s: %s", batch_id, exc)
            record["last_error"] = str(exc)
            remaining.append(record)
            continue

        status = maybe_get_attr(batch_info, "status")
        record["last_status"] = status
        record["last_checked_at"] = iso_timestamp()
        logger.info("Batch %s status: %s", batch_id, status)

        output_file_id = maybe_get_attr(batch_info, "output_file_id")
        error_file_id = maybe_get_attr(batch_info, "error_file_id")

        output_path = Path(record.get("output_path", responses_dir / f"{batch_id}.jsonl"))
        if status == "completed":
            if output_file_id:
                content = client.download_file(output_file_id)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
                logger.info("Downloaded batch %s output to %s.", batch_id, output_path)

            if error_file_id:
                error_path = output_path.with_suffix(".errors.jsonl")
                error_content = client.download_file(error_file_id)
                error_path.write_bytes(error_content)
                logger.info("Saved batch %s errors to %s.", batch_id, error_path)
        else:
            if status in {"failed", "expired", "cancelled"} and error_file_id:
                error_path = Path(record.get("output_path", responses_dir / f"{batch_id}.jsonl")).with_suffix(".errors.jsonl")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_content = client.download_file(error_file_id)
                error_path.write_bytes(error_content)
                logger.warning("Batch %s reported %s. Saved error log to %s.", batch_id, status, error_path)
            else:
                remaining.append(record)

    save_requests(requests_file, remaining)
    logger.info("Updated %s with %d pending batch(es).", requests_file, len(remaining))


def parse_metadata(metadata_argument: str | None) -> dict[str, Any] | None:
    if not metadata_argument:
        return None
    try:
        parsed = json.loads(metadata_argument)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--metadata must encode a JSON object.")

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit and retrieve gpt51 batch API jobs.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider label used for batch tracking and default response paths.")
    parser.add_argument("--api-key", help="API key. Overrides provider-specific environment variables.")
    parser.add_argument("--api-base", help="Override the API base URL (default: https://api.openai.com/v1).")
    parser.add_argument("--beta-header", help="Optional value for the batch beta header.")
    parser.add_argument("--request", type=Path, help="Path to a JSONL file to submit to the Batch API.")
    parser.add_argument("--receive", action="store_true", help="Attempt to download completed batches listed in the requests file.")
    parser.add_argument("--requests-file", type=Path, help="File tracking outstanding batch IDs. Defaults to gpt-querying/requests.<provider>.jsonl.")
    parser.add_argument("--responses-dir", type=Path, help="Directory to store downloaded batch outputs. Defaults to ../responses/<provider>.")
    parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW, help="Completion window requested for new batches.")
    parser.add_argument("--metadata", help="Optional JSON object to attach as batch metadata.")

    return parser


def resolve_api_key(provider: str, cli_key: str | None) -> str:
    load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=False)

    prefix = provider_env_prefix(provider)
    api_key = (
        cli_key
        or os.getenv(f"{prefix}_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            f"API key not provided for provider '{provider}'. "
            f"Use --api-key or set {prefix}_API_KEY, LLM_API_KEY, or OPENAI_API_KEY in the environment/.env."
        )

    return api_key


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.request and not args.receive:
        parser.error("Specify at least one action: --request PATH and/or --receive.")

    requests_file = args.requests_file or default_requests_file(args.provider)
    responses_dir = args.responses_dir or default_responses_dir(args.provider)
    api_key = resolve_api_key(args.provider, args.api_key)
    metadata = parse_metadata(args.metadata)

    provider_prefix = provider_env_prefix(args.provider)
    api_base = (
        args.api_base
        or os.getenv(f"{provider_prefix}_API_BASE")
        or os.getenv("LLM_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_API_BASE
    )
    beta_header = (
        args.beta_header
        or os.getenv(f"{provider_prefix}_BETA_HEADER")
        or os.getenv("LLM_BETA_HEADER")
        or os.getenv("OPENAI_BETA_HEADER")
    )
    client = CompatibleBatchClient(api_key=api_key, api_base=api_base, beta_header=beta_header)

    if args.request:
        submit_request(
            client=client,
            jsonl_path=args.request,
            requests_file=requests_file,
            responses_dir=responses_dir,
            completion_window=args.completion_window,
            metadata=metadata,
            provider=args.provider,
        )

    if args.receive:
        receive_batches(
            client=client,
            requests_file=requests_file,
            responses_dir=responses_dir,
        )


if __name__ == "__main__":
    main()
