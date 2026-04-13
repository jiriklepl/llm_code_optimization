#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
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
SUPPORTED_PROVIDERS = ("gpt51", "gemma")
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_GPT51_API_BASE = "https://api.openai.com/v1"
DEFAULT_GEMMA_API_BASE = "http://bw01:8000/v1"


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


def append_jsonl(filepath: Path, record: dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def provider_env_prefix(provider: str) -> str:
    return provider.upper().replace("-", "_")


def get_supported_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}.")
    return normalized


def env_first(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def default_requests_file(provider: str) -> Path:
    return SCRIPT_DIR / f"requests.{provider}.jsonl"


def default_responses_dir(provider: str) -> Path:
    responses_dir = SCRIPT_DIR.parent / "responses" / provider
    try:
        return Path("..") / responses_dir.relative_to(CURRENT_DIR.parent)
    except ValueError:
        return responses_dir


def resolve_output_path(jsonl_path: Path, responses_dir: Path, provider: str) -> Path:
    jsonl_path = jsonl_path.resolve()
    parent_name = jsonl_path.parent.name
    if jsonl_path.parent.parent.name == provider and jsonl_path.parent.parent.parent.name == "requests":
        responses_dir = responses_dir / parent_name

    responses_dir.mkdir(parents=True, exist_ok=True)
    return responses_dir / f"{jsonl_path.stem}.jsonl"


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


def parse_json_response(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def join_api_url(api_base: str, endpoint: str) -> str:
    parsed_base = urllib.parse.urlsplit(api_base)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError(f"Invalid API base URL: {api_base!r}")

    base_path = parsed_base.path.rstrip("/")
    endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base_path and (endpoint_path == base_path or endpoint_path.startswith(base_path + "/")):
        joined_path = endpoint_path
    else:
        joined_path = f"{base_path}{endpoint_path}"

    return urllib.parse.urlunsplit(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            joined_path,
            parsed_base.query,
            "",
        )
    )


def resolve_api_key(provider: str, cli_key: str | None, *, required: bool) -> str | None:
    prefix = provider_env_prefix(provider)
    env_keys = [f"{prefix}_API_KEY", "LLM_API_KEY"]
    if provider == "gpt51":
        env_keys.append("OPENAI_API_KEY")

    api_key = cli_key or env_first(*env_keys)
    if required and not api_key:
        key_list = ", ".join(env_keys)
        raise SystemExit(
            f"API key not provided for provider '{provider}'. "
            f"Use --api-key or set one of: {key_list}."
        )
    return api_key


def resolve_api_base(provider: str, cli_api_base: str | None) -> str:
    if provider == "gpt51":
        return cli_api_base or env_first("GPT51_API_BASE", "LLM_API_BASE", "OPENAI_API_BASE") or DEFAULT_GPT51_API_BASE
    if provider == "gemma":
        return cli_api_base or env_first("GEMMA_API_BASE", "LLM_API_BASE") or DEFAULT_GEMMA_API_BASE
    raise AssertionError(f"Unhandled provider {provider!r}")


def resolve_beta_header(provider: str, cli_beta_header: str | None) -> str | None:
    if provider != "gpt51":
        return None
    return cli_beta_header or env_first("GPT51_BETA_HEADER", "LLM_BETA_HEADER", "OPENAI_BETA_HEADER")


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


@dataclass
class GemmaRequestError(Exception):
    message: str
    error_type: str = "request_error"
    status_code: int | None = None
    body: Any | None = None
    request_id: str | None = None

    def __str__(self) -> str:
        return self.message


class Gpt51BatchClient:
    def __init__(self, api_key: str, api_base: str | None = DEFAULT_GPT51_API_BASE, beta_header: str | None = None) -> None:
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


def submit_gpt51_request(
    client: Gpt51BatchClient,
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

    output_path = resolve_output_path(jsonl_path, responses_dir, provider)

    logger.info("Uploading %s to %s Files API.", jsonl_path.resolve(), provider)
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

    record = BatchRecord(
        batch_id=batch_id,
        input_path=str(jsonl_path.resolve()),
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


def receive_gpt51_batches(
    client: Gpt51BatchClient,
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


def post_gemma_request(
    api_base: str,
    endpoint: str,
    body: dict[str, Any],
    api_key: str | None,
) -> tuple[Any, int, str | None]:
    url = join_api_url(api_base, endpoint)
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8", errors="replace")
            parsed_response = parse_json_response(response_body)
            if not isinstance(parsed_response, dict):
                raise GemmaRequestError(
                    f"Gemma response from {url} was not valid JSON.",
                    error_type="invalid_json",
                    status_code=status_code,
                    body=parsed_response,
                    request_id=response.headers.get("x-request-id"),
                )
            return parsed_response, status_code, response.headers.get("x-request-id")
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise GemmaRequestError(
            f"Gemma request to {url} failed with HTTP {exc.code}.",
            status_code=exc.code,
            body=parse_json_response(response_body),
            request_id=exc.headers.get("x-request-id"),
        ) from exc
    except urllib.error.URLError as exc:
        raise GemmaRequestError(
            f"Gemma request to {url} failed: {exc.reason}.",
            error_type="connection_error",
        ) from exc


def normalize_gemma_response(custom_id: str, response_body: dict[str, Any], status_code: int, request_id: str | None) -> dict[str, Any]:
    return {
        "id": response_body.get("id", f"gemma_req_{custom_id}"),
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": request_id,
            "body": response_body,
        },
        "error": None,
    }


def write_gemma_error(error_path: Path, custom_id: str, request_record: dict[str, Any], error: GemmaRequestError) -> None:
    append_jsonl(
        error_path,
        {
            "custom_id": custom_id,
            "error": {
                "type": error.error_type,
                "message": str(error),
                "status_code": error.status_code,
                "request_id": error.request_id,
                "body": error.body,
            },
            "request": request_record,
        },
    )


def submit_gemma_request(
    jsonl_path: Path,
    responses_dir: Path,
    api_base: str,
    api_key: str | None,
    provider: str,
) -> Path:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Request file not found: {jsonl_path}")

    output_path = resolve_output_path(jsonl_path, responses_dir, provider)
    error_path = output_path.with_suffix(".errors.jsonl")
    if error_path.exists():
        error_path.unlink()

    successes = 0
    failures = 0

    with jsonl_path.open("r", encoding="utf-8") as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line_number, raw_line in enumerate(input_handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                request_record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures += 1
                append_jsonl(
                    error_path,
                    {
                        "custom_id": f"line-{line_number:05d}",
                        "error": {
                            "type": "invalid_request_json",
                            "message": f"Line {line_number} is not valid JSON: {exc}",
                            "status_code": None,
                            "request_id": None,
                            "body": raw_line,
                        },
                    },
                )
                continue

            custom_id = request_record.get("custom_id", f"line-{line_number:05d}")
            endpoint = request_record.get("url")
            body = request_record.get("body")
            if not endpoint or not isinstance(body, dict):
                failures += 1
                write_gemma_error(
                    error_path,
                    custom_id,
                    request_record,
                    GemmaRequestError(
                        "Gemma request line must include 'url' and a JSON-object 'body'.",
                        error_type="invalid_request_shape",
                    ),
                )
                continue

            try:
                response_body, status_code, request_id = post_gemma_request(api_base, endpoint, body, api_key)
            except GemmaRequestError as exc:
                failures += 1
                write_gemma_error(error_path, custom_id, request_record, exc)
                continue

            json.dump(normalize_gemma_response(custom_id, response_body, status_code, request_id), output_handle)
            output_handle.write("\n")
            output_handle.flush()
            successes += 1

    logger.info("Stored %d gemma response(s) in %s.", successes, output_path)
    if failures:
        logger.error("Gemma processing finished with %d error(s). See %s.", failures, error_path)
        raise SystemExit(1)
    return output_path


def receive_gemma_requests(requests_file: Path) -> None:
    logger.info("No pending batch requests in %s.", requests_file)


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
    parser = argparse.ArgumentParser(description="Submit and retrieve provider-specific query jobs.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider label used for request execution and default response paths.")
    parser.add_argument("--api-key", help="API key. Overrides provider-specific environment variables when supported.")
    parser.add_argument("--api-base", help="Override the API base URL. Defaults depend on the provider.")
    parser.add_argument("--beta-header", help="Optional value for provider-specific beta headers (used by gpt51).")
    parser.add_argument("--request", type=Path, help="Path to a JSONL file to submit.")
    parser.add_argument("--receive", action="store_true", help="Receive or poll for responses for the selected provider.")
    parser.add_argument("--requests-file", type=Path, help="File tracking outstanding batch IDs. Defaults to gpt-querying/requests.<provider>.jsonl.")
    parser.add_argument("--responses-dir", type=Path, help="Directory to store provider outputs. Defaults to ../responses/<provider>.")
    parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW, help="Completion window requested for new gpt51 batches.")
    parser.add_argument("--metadata", help="Optional JSON object to attach as gpt51 batch metadata.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.request and not args.receive:
        parser.error("Specify at least one action: --request PATH and/or --receive.")

    load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=False)

    try:
        provider = get_supported_provider(args.provider)
        metadata = parse_metadata(args.metadata)
        requests_file = args.requests_file or default_requests_file(provider)
        responses_dir = args.responses_dir or default_responses_dir(provider)
        api_base = resolve_api_base(provider, args.api_base)
    except ValueError as exc:
        parser.error(str(exc))

    if provider == "gpt51":
        api_key = resolve_api_key(provider, args.api_key, required=True)
        assert api_key is not None
        client = Gpt51BatchClient(
            api_key=api_key,
            api_base=api_base,
            beta_header=resolve_beta_header(provider, args.beta_header),
        )

        if args.request:
            submit_gpt51_request(
                client=client,
                jsonl_path=args.request,
                requests_file=requests_file,
                responses_dir=responses_dir,
                completion_window=args.completion_window,
                metadata=metadata,
                provider=provider,
            )

        if args.receive:
            receive_gpt51_batches(
                client=client,
                requests_file=requests_file,
                responses_dir=responses_dir,
            )
        return

    api_key = resolve_api_key(provider, args.api_key, required=False)

    if args.request:
        submit_gemma_request(
            jsonl_path=args.request,
            responses_dir=responses_dir,
            api_base=api_base,
            api_key=api_key,
            provider=provider,
        )

    if args.receive:
        receive_gemma_requests(requests_file)


if __name__ == "__main__":
    main()
