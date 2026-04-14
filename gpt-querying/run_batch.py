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
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import APIError, OpenAI

from providers import (
    DEFAULT_ANTHROPIC_BETA_HEADER,
    DEFAULT_ANTHROPIC_VERSION,
    DEFAULT_PROVIDER_API_BASES,
    DEFAULT_PROVIDER_NAME,
    normalize_provider_name,
    provider_env_prefix,
)

logger = logging.getLogger("run_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CURRENT_DIR = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLETION_WINDOW = "24h"


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


def default_requests_file(provider: str) -> Path:
    canonical_provider = normalize_provider_name(provider)
    return SCRIPT_DIR / f"requests.{canonical_provider}.jsonl"


def default_responses_dir(provider: str) -> Path:
    canonical_provider = normalize_provider_name(provider)
    responses_dir = SCRIPT_DIR.parent / "responses" / canonical_provider
    try:
        return Path("..") / responses_dir.relative_to(CURRENT_DIR.parent)
    except ValueError:
        return responses_dir


def get_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    sentinel = object()
    value = getattr(obj, key, sentinel)
    if value is not sentinel:
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


def normalize_download_content(content: Any) -> bytes:
    if hasattr(content, "read"):
        data = content.read()
        if hasattr(content, "close"):
            content.close()
        return data
    if hasattr(content, "content"):
        return content.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    raise TypeError(f"Unsupported downloaded content type: {type(content)!r}")


@dataclass
class BatchRecord:
    batch_id: str
    input_path: str
    provider: str
    submitted_at: str
    output_path: str
    endpoint: str | None = None
    completion_window: str | None = None
    metadata: dict[str, Any] | None = None
    results_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_id": self.batch_id,
            "input_path": self.input_path,
            "provider": self.provider,
            "submitted_at": self.submitted_at,
            "output_path": self.output_path,
        }
        if self.endpoint is not None:
            data["endpoint"] = self.endpoint
        if self.completion_window is not None:
            data["completion_window"] = self.completion_window
        if self.metadata is not None:
            data["metadata"] = self.metadata
        if self.results_url is not None:
            data["results_url"] = self.results_url
        return data


class OpenAICompatibleBatchClient:
    def __init__(self, api_key: str, api_base: str, beta_header: str | None = None) -> None:
        default_headers: dict[str, str] | None = None
        if beta_header:
            default_headers = {"OpenAI-Beta": beta_header}

        init_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": api_base}
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
        return normalize_download_content(self._client.files.content(file_id=file_id))


def build_genai_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def build_google_upload_file_config(display_name: str):
    from google.genai import types

    return types.UploadFileConfig(display_name=display_name, mime_type="jsonl")


class GeminiBatchClient(OpenAICompatibleBatchClient):
    def __init__(self, api_key: str, api_base: str, beta_header: str | None = None) -> None:
        super().__init__(api_key=api_key, api_base=api_base, beta_header=beta_header)
        self._file_client = build_genai_client(api_key)

    def upload_batch_file(self, filepath: Path):
        config = build_google_upload_file_config(filepath.stem)
        return self._file_client.files.upload(file=str(filepath), config=config)

    def download_file(self, file_id: str) -> bytes:
        return normalize_download_content(self._file_client.files.download(file=file_id))


class AnthropicBatchClient:
    def __init__(self, api_key: str, api_base: str, anthropic_version: str, beta_header: str | None = None) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/") + "/"
        self.anthropic_version = anthropic_version
        self.beta_header = beta_header

    def _make_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return urljoin(self.api_base, path_or_url.lstrip("/"))

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        if self.beta_header:
            headers["anthropic-beta"] = self.beta_header
        return headers

    def _request_json(self, method: str, path_or_url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self._make_url(path_or_url), data=data, headers=self._headers(), method=method)

        try:
            with urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API request failed with status {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Anthropic API request failed: {exc}") from exc

    def _request_bytes(self, method: str, path_or_url: str) -> bytes:
        request = Request(self._make_url(path_or_url), headers=self._headers(), method=method)

        try:
            with urlopen(request) as response:
                return response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API download failed with status {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Anthropic API download failed: {exc}") from exc

    def create_batch(self, requests_payload: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request_json("POST", "/messages/batches", {"requests": requests_payload})

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/messages/batches/{batch_id}")

    def download_results(self, results_url: str) -> bytes:
        return self._request_bytes("GET", results_url)


def response_dir_for_request(jsonl_path: Path, responses_dir: Path, provider: str) -> Path:
    canonical_provider = normalize_provider_name(provider)
    parent_name = jsonl_path.parent.name
    if jsonl_path.parent.parent.name == canonical_provider and jsonl_path.parent.parent.parent.name == "requests":
        return responses_dir / parent_name
    return responses_dir


def submit_openai_compatible_request(
    client: OpenAICompatibleBatchClient,
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
    responses_dir = response_dir_for_request(jsonl_path, responses_dir, provider)

    logger.info("Uploading %s to %s batch files API.", jsonl_path, provider)
    file_response = client.upload_batch_file(jsonl_path)
    input_file_id = maybe_get_attr(file_response, "name") or get_attr(file_response, "id")
    logger.info("Uploaded file. input_file_id=%s", input_file_id)

    logger.info("Creating batch for endpoint %s.", endpoint)
    batch_response = client.create_batch(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
    )
    batch_id = get_attr(batch_response, "id")
    status = maybe_get_attr(batch_response, "status")
    logger.info("Created batch %s with status %s.", batch_id, status)

    responses_dir.mkdir(parents=True, exist_ok=True)
    output_path = responses_dir / f"{jsonl_path.stem}.jsonl"

    record = BatchRecord(
        batch_id=batch_id,
        input_path=str(jsonl_path),
        provider=normalize_provider_name(provider),
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


def receive_openai_compatible_batches(
    client: OpenAICompatibleBatchClient,
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
            continue

        if status in {"failed", "expired", "cancelled"} and error_file_id:
            error_path = output_path.with_suffix(".errors.jsonl")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_content = client.download_file(error_file_id)
            error_path.write_bytes(error_content)
            logger.warning("Batch %s reported %s. Saved error log to %s.", batch_id, status, error_path)
            continue

        remaining.append(record)

    save_requests(requests_file, remaining)
    logger.info("Updated %s with %d pending batch(es).", requests_file, len(remaining))


def submit_anthropic_request(
    client: AnthropicBatchClient,
    jsonl_path: Path,
    requests_file: Path,
    responses_dir: Path,
    completion_window: str,
    metadata: dict[str, Any] | None,
    provider: str,
) -> BatchRecord:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Request file not found: {jsonl_path}")

    request_lines = load_requests(jsonl_path)
    if not request_lines:
        raise ValueError(f"Request file {jsonl_path} does not contain any valid anthropic batch entries.")

    if metadata:
        logger.warning("Anthropic message batches do not support top-level metadata in this workflow; metadata will only be stored locally.")

    jsonl_path = jsonl_path.resolve()
    responses_dir = response_dir_for_request(jsonl_path, responses_dir, provider)

    logger.info("Creating anthropic message batch from %s.", jsonl_path)
    batch_response = client.create_batch(request_lines)
    batch_id = batch_response["id"]
    status = batch_response.get("processing_status")
    logger.info("Created anthropic batch %s with status %s.", batch_id, status)

    responses_dir.mkdir(parents=True, exist_ok=True)
    output_path = responses_dir / f"{jsonl_path.stem}.jsonl"

    record = BatchRecord(
        batch_id=batch_id,
        input_path=str(jsonl_path),
        provider=normalize_provider_name(provider),
        endpoint="/v1/messages/batches",
        completion_window=completion_window,
        submitted_at=iso_timestamp(),
        output_path=str(output_path),
        metadata=metadata,
        results_url=batch_response.get("results_url"),
    )

    existing = load_requests(requests_file)
    existing.append(record.to_dict())
    save_requests(requests_file, existing)

    logger.info("Stored anthropic batch %s in %s.", batch_id, requests_file)
    return record


def receive_anthropic_batches(
    client: AnthropicBatchClient,
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
        except Exception as exc:
            logger.error("Failed to retrieve anthropic batch %s: %s", batch_id, exc)
            record["last_error"] = str(exc)
            remaining.append(record)
            continue

        status = batch_info.get("processing_status")
        record["last_status"] = status
        record["last_checked_at"] = iso_timestamp()
        logger.info("Anthropic batch %s status: %s", batch_id, status)

        output_path = Path(record.get("output_path", responses_dir / f"{batch_id}.jsonl"))
        results_url = batch_info.get("results_url") or record.get("results_url")

        if status == "ended":
            if results_url:
                content = client.download_results(results_url)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
                logger.info("Downloaded anthropic batch %s output to %s.", batch_id, output_path)
            else:
                status_path = output_path.with_suffix(".errors.json")
                status_path.write_text(json.dumps(batch_info, indent=2, sort_keys=True), encoding="utf-8")
                logger.warning("Anthropic batch %s ended without results_url. Saved status to %s.", batch_id, status_path)
            continue

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
    parser = argparse.ArgumentParser(description="Submit and retrieve provider batch API jobs.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER_NAME, help="Provider label used for batch tracking and default response paths.")
    parser.add_argument("--api-key", help="API key. Overrides provider-specific environment variables.")
    parser.add_argument("--api-base", help="Override the API base URL.")
    parser.add_argument("--beta-header", help="Optional provider-specific beta header. Anthropic defaults to output-300k-2026-03-24.")
    parser.add_argument("--anthropic-version", help="Anthropic API version header. Defaults to GPT_QUERYING_ANTHROPIC_VERSION or ANTHROPIC_VERSION.")
    parser.add_argument("--request", type=Path, help="Path to a JSONL file to submit to the batch API.")
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


def resolve_api_base(provider: str, cli_api_base: str | None) -> str:
    prefix = provider_env_prefix(provider)
    return (
        cli_api_base
        or os.getenv(f"{prefix}_API_BASE")
        or os.getenv("LLM_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_PROVIDER_API_BASES[provider]
    )


def resolve_beta_header(provider: str, cli_beta_header: str | None) -> str | None:
    prefix = provider_env_prefix(provider)
    if normalize_provider_name(provider) == "anthropic":
        return (
            cli_beta_header
            or os.getenv(f"{prefix}_BETA_HEADER")
            or os.getenv("LLM_BETA_HEADER")
            or DEFAULT_ANTHROPIC_BETA_HEADER
        )

    return (
        cli_beta_header
        or os.getenv(f"{prefix}_BETA_HEADER")
        or os.getenv("LLM_BETA_HEADER")
        or os.getenv("OPENAI_BETA_HEADER")
    )


def resolve_anthropic_version(cli_anthropic_version: str | None) -> str:
    return (
        cli_anthropic_version
        or os.getenv("GPT_QUERYING_ANTHROPIC_VERSION")
        or os.getenv("ANTHROPIC_VERSION")
        or DEFAULT_ANTHROPIC_VERSION
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.request and not args.receive:
        parser.error("Specify at least one action: --request PATH and/or --receive.")

    provider = normalize_provider_name(args.provider)
    requests_file = args.requests_file or default_requests_file(provider)
    responses_dir = args.responses_dir or default_responses_dir(provider)
    api_key = resolve_api_key(provider, args.api_key)
    api_base = resolve_api_base(provider, args.api_base)
    beta_header = resolve_beta_header(provider, args.beta_header)
    metadata = parse_metadata(args.metadata)

    if provider == "anthropic":
        client = AnthropicBatchClient(
            api_key=api_key,
            api_base=api_base,
            anthropic_version=resolve_anthropic_version(args.anthropic_version),
            beta_header=beta_header,
        )

        if args.request:
            submit_anthropic_request(
                client=client,
                jsonl_path=args.request,
                requests_file=requests_file,
                responses_dir=responses_dir,
                completion_window=args.completion_window,
                metadata=metadata,
                provider=provider,
            )

        if args.receive:
            receive_anthropic_batches(
                client=client,
                requests_file=requests_file,
                responses_dir=responses_dir,
            )
        return

    if provider == "gemini":
        client: OpenAICompatibleBatchClient = GeminiBatchClient(
            api_key=api_key,
            api_base=api_base,
            beta_header=beta_header,
        )
    else:
        client = OpenAICompatibleBatchClient(
            api_key=api_key,
            api_base=api_base,
            beta_header=beta_header,
        )

    if args.request:
        submit_openai_compatible_request(
            client=client,
            jsonl_path=args.request,
            requests_file=requests_file,
            responses_dir=responses_dir,
            completion_window=args.completion_window,
            metadata=metadata,
            provider=provider,
        )

    if args.receive:
        receive_openai_compatible_batches(
            client=client,
            requests_file=requests_file,
            responses_dir=responses_dir,
        )


if __name__ == "__main__":
    main()
