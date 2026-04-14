#!/usr/bin/env python3

from __future__ import annotations

import argparse
from argparse import ArgumentParser
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Generator, List

from colorama import Fore, Style
from munch import munchify

from providers import (
    DEFAULT_PROVIDER_MODELS,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VERBOSITY,
    anthropic_prefers_adaptive_thinking,
    env_int,
    normalize_provider_name,
    resolve_anthropic_effort,
    resolve_anthropic_max_tokens,
    resolve_anthropic_thinking_budget,
    resolve_gemini_reasoning_effort,
    resolve_gemini_include_thoughts,
    resolve_gemini_thinking_budget,
    resolve_provider_model,
    resolve_reasoning_effort,
    resolve_verbosity,
)

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


DEFAULT_REPETITIONS = env_int("GPT_QUERYING_REPETITIONS", "GPT_QUERYING_REPS", default=5)
TEST_BENCHMARKS = ("2mm", "floyd-warshall", "heat-3d", "gemm")
CATEGORIES = ["linear-algebra", "datamining", "stencil", "medley", "full", "test"]


def object_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def first_present(*values: Any) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


@dataclass
class ParsedBatchResult:
    custom_id: str
    response_body: Any | None = None
    error_payload: dict[str, Any] | None = None


class BatchProvider(ABC):
    method: str = "POST"
    url: str = "/"

    def format_custom_id(self, base: str, repetition: int) -> str:
        return f"{base}-{repetition:02d}"

    def create_batch_entry(self, custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": self.method,
            "url": self.url,
            "body": body,
        }

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        return None

    @abstractmethod
    def parse_batch_result(self, batch_result: dict[str, Any]) -> ParsedBatchResult:
        ...

    @abstractmethod
    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        ...

    @abstractmethod
    def extract_answer(self, response: Any) -> str:
        ...


@dataclass
class OpenAICompatibleBatchProvider(BatchProvider):
    name: str
    model: str
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    url: str = "/v1/chat/completions"
    method: str = "POST"

    def parse_batch_result(self, batch_result: dict[str, Any]) -> ParsedBatchResult:
        custom_id = batch_result["custom_id"]
        response = batch_result.get("response")
        if not isinstance(response, dict):
            return ParsedBatchResult(
                custom_id=custom_id,
                error_payload={
                    "message": "Batch result is missing a response payload.",
                    "raw": batch_result,
                },
            )

        status_code = response.get("status_code")
        body = response.get("body")
        if status_code != 200 or body is None:
            return ParsedBatchResult(
                custom_id=custom_id,
                error_payload={
                    "status_code": status_code,
                    "body": body,
                    "error": batch_result.get("error"),
                },
            )

        return ParsedBatchResult(custom_id=custom_id, response_body=munchify(body))

    def extract_answer(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                object_get(item, "text")
                for item in content
                if object_get(item, "text")
            ]
            return "\n".join(text_parts)
        return str(content)


@dataclass
class OpenAIBatchProvider(OpenAICompatibleBatchProvider):
    verbosity: str = DEFAULT_VERBOSITY
    cost_input: float = 0.625 / 1_000_000
    cost_cached_input: float = 0.0625 / 1_000_000
    cost_output: float = 5 / 1_000_000

    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
            "messages": messages,
        }

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        try:
            prompt = response.usage.prompt_tokens
            prompt_cached = response.usage.prompt_tokens_details.cached_tokens
            completion = response.usage.completion_tokens
            reasoning = response.usage.completion_tokens_details.reasoning_tokens
        except AttributeError:
            logger.debug("Batch response missing usage details; skipping cost estimation.")
            return

        prompt_uncached = prompt - prompt_cached
        output = completion - reasoning

        prompt_uncached_cost = prompt_uncached * self.cost_input
        prompt_cached_cost = prompt_cached * self.cost_cached_input
        cost_output = output * self.cost_output
        cost_reasoning = reasoning * self.cost_output

        total_cost = prompt_uncached_cost + prompt_cached_cost + cost_output + cost_reasoning

        costs = {
            "prompt_uncached_tokens": prompt_uncached,
            "prompt_uncached_cost": prompt_uncached_cost,
            "prompt_cached_tokens": prompt_cached,
            "prompt_cached_cost": prompt_cached_cost,
            "output_tokens": output,
            "cost_output": cost_output,
            "reasoning_tokens": reasoning,
            "cost_reasoning": cost_reasoning,
            "total_cost": total_cost,
        }
        if time_taken:
            costs["time_taken_in_seconds"] = time_taken

        costs_log = json.dumps(costs, indent=2)

        logger.info(f"Estimated cost (USD):\n{Fore.YELLOW}{costs_log}{Style.RESET_ALL}")

        (output_folder / "costs.json").write_text(costs_log, encoding="utf-8")


@dataclass
class GeminiBatchProvider(OpenAICompatibleBatchProvider):
    gemini_thinking_budget: int | None = None
    gemini_include_thoughts: bool = False

    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        thinking_config: dict[str, Any] = {}

        if self.gemini_thinking_budget is not None:
            thinking_config["thinking_budget"] = self.gemini_thinking_budget
        if self.gemini_include_thoughts:
            thinking_config["include_thoughts"] = True

        if not thinking_config:
            gemini_reasoning_effort = resolve_gemini_reasoning_effort(self.model, self.reasoning_effort)
            if gemini_reasoning_effort is not None:
                request["reasoning_effort"] = gemini_reasoning_effort

        if thinking_config:
            request["extra_body"] = {
                "google": {
                    "thinking_config": thinking_config,
                }
            }

        return request

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        usage = first_present(
            object_get(response, "usageMetadata"),
            object_get(response, "usage_metadata"),
            object_get(response, "usage"),
        )
        if usage is None:
            logger.debug("Gemini batch response missing usage details; skipping cost estimation.")
            return

        prompt = first_present(
            object_get(usage, "promptTokenCount"),
            object_get(usage, "prompt_token_count"),
            object_get(usage, "prompt_tokens"),
        )
        prompt_cached = first_present(
            object_get(usage, "cachedContentTokenCount"),
            object_get(usage, "cached_content_token_count"),
            object_get(object_get(usage, "prompt_tokens_details"), "cached_tokens"),
            0,
        )
        reasoning = first_present(
            object_get(usage, "thoughtsTokenCount"),
            object_get(usage, "thoughts_token_count"),
            object_get(object_get(usage, "completion_tokens_details"), "reasoning_tokens"),
            0,
        )
        completion = first_present(
            object_get(usage, "completion_tokens"),
            object_get(usage, "totalTokenCount"),
            object_get(usage, "total_token_count"),
        )
        output = first_present(
            object_get(usage, "candidatesTokenCount"),
            object_get(usage, "candidates_token_count"),
        )

        if output is None and completion is not None:
            output = max(int(completion) - int(reasoning), 0)
        if prompt is None or output is None:
            logger.debug("Gemini batch response did not expose token counts in a supported shape.")
            return

        prompt = int(prompt)
        prompt_cached = int(prompt_cached or 0)
        reasoning = int(reasoning or 0)
        output = int(output)

        prompt_uncached = max(prompt - prompt_cached, 0)
        low_tier = prompt <= 200_000
        cost_input = (0.625 if low_tier else 1.25) / 1_000_000
        cost_output = (5 if low_tier else 7.5) / 1_000_000

        prompt_uncached_cost = prompt_uncached * cost_input
        prompt_cached_cost = 0.0
        output_cost = output * cost_output
        reasoning_cost = reasoning * cost_output

        costs = {
            "prompt_uncached_tokens": prompt_uncached,
            "prompt_uncached_cost": prompt_uncached_cost,
            "prompt_cached_tokens": prompt_cached,
            "prompt_cached_cost": prompt_cached_cost,
            "cached_cost_estimate_unavailable": prompt_cached > 0,
            "output_tokens": output,
            "cost_output": output_cost,
            "reasoning_tokens": reasoning,
            "cost_reasoning": reasoning_cost,
            "total_cost": prompt_uncached_cost + prompt_cached_cost + output_cost + reasoning_cost,
        }
        if time_taken:
            costs["time_taken_in_seconds"] = time_taken

        costs_log = json.dumps(costs, indent=2)
        logger.info(f"Estimated cost (USD):\n{Fore.YELLOW}{costs_log}{Style.RESET_ALL}")
        (output_folder / "costs.json").write_text(costs_log, encoding="utf-8")


@dataclass
class AnthropicBatchProvider(BatchProvider):
    model: str
    max_tokens: int
    thinking_budget: int | None = None
    effort: str | None = None
    adaptive_thinking: bool = False
    method: str = "POST"
    url: str = "/v1/messages"

    def create_batch_entry(self, custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "params": body,
        }

    def parse_batch_result(self, batch_result: dict[str, Any]) -> ParsedBatchResult:
        custom_id = batch_result.get("custom_id") or batch_result.get("id")
        if not custom_id:
            raise KeyError("Anthropic batch result is missing custom_id/id.")

        result = batch_result.get("result")
        if not isinstance(result, dict):
            return ParsedBatchResult(
                custom_id=custom_id,
                error_payload={
                    "message": "Anthropic batch result is missing a result payload.",
                    "raw": batch_result,
                },
            )

        result_type = result.get("type")
        if result_type == "succeeded":
            return ParsedBatchResult(
                custom_id=custom_id,
                response_body=munchify(result.get("message", {})),
            )

        return ParsedBatchResult(custom_id=custom_id, error_payload=result)

    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
            else:
                anthropic_messages.append({"role": role, "content": content})

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)
        if self.effort is not None:
            request["output_config"] = {"effort": self.effort}
        if self.thinking_budget is not None:
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        elif self.adaptive_thinking:
            request["thinking"] = {"type": "adaptive"}

        return request

    def extract_answer(self, response: Any) -> str:
        content = object_get(response, "content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        text_parts = [
            object_get(block, "text")
            for block in content
            if object_get(block, "type") == "text" and object_get(block, "text")
        ]
        return "\n".join(text_parts).strip()

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        usage = object_get(response, "usage")
        if usage is None:
            logger.debug("Anthropic batch response missing usage details; skipping cost estimation.")
            return

        prompt_uncached = int(object_get(usage, "input_tokens", 0) or 0)
        output = int(object_get(usage, "output_tokens", 0) or 0)
        prompt_uncached_cost = prompt_uncached * (7.5 / 1_000_000)
        output_cost = output * (37.5 / 1_000_000)

        costs = {
            "prompt_uncached_tokens": prompt_uncached,
            "prompt_uncached_cost": prompt_uncached_cost,
            "prompt_cached_tokens": 0,
            "prompt_cached_cost": 0.0,
            "output_tokens": output,
            "cost_output": output_cost,
            "reasoning_tokens": 0,
            "cost_reasoning": 0.0,
            "thinking_token_split_unavailable": True,
            "total_cost": prompt_uncached_cost + output_cost,
        }
        if time_taken:
            costs["time_taken_in_seconds"] = time_taken

        costs_log = json.dumps(costs, indent=2)
        logger.info(f"Estimated cost (USD):\n{Fore.YELLOW}{costs_log}{Style.RESET_ALL}")
        (output_folder / "costs.json").write_text(costs_log, encoding="utf-8")


def build_batch_provider(
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model: str | None = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    verbosity: str = DEFAULT_VERBOSITY,
    gemini_thinking_budget: int | None = None,
    gemini_include_thoughts: bool = False,
    anthropic_thinking_budget: int | None = None,
    anthropic_max_tokens: int | None = None,
) -> BatchProvider:
    canonical_provider = normalize_provider_name(provider_name)
    resolved_model = model or DEFAULT_PROVIDER_MODELS[canonical_provider]

    if canonical_provider == "openai":
        return OpenAIBatchProvider(
            name=canonical_provider,
            model=resolved_model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
    if canonical_provider == "gemini":
        return GeminiBatchProvider(
            name=canonical_provider,
            model=resolved_model,
            reasoning_effort=reasoning_effort,
            gemini_thinking_budget=gemini_thinking_budget,
            gemini_include_thoughts=gemini_include_thoughts,
        )
    if canonical_provider == "anthropic":
        anthropic_effort = resolve_anthropic_effort(resolved_model, reasoning_effort)
        return AnthropicBatchProvider(
            model=resolved_model,
            max_tokens=anthropic_max_tokens if anthropic_max_tokens is not None else resolve_anthropic_max_tokens(
                thinking_budget=anthropic_thinking_budget,
            ),
            thinking_budget=anthropic_thinking_budget,
            effort=anthropic_effort,
            adaptive_thinking=anthropic_effort is not None and anthropic_thinking_budget is None and anthropic_prefers_adaptive_thinking(resolved_model),
        )

    raise SystemExit(f"Unsupported provider: {provider_name}")


DEFAULT_PROVIDER_MODEL = resolve_provider_model(DEFAULT_PROVIDER_NAME)
DEFAULT_ANTHROPIC_THINKING_BUDGET = resolve_anthropic_thinking_budget(
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    model=DEFAULT_PROVIDER_MODEL if DEFAULT_PROVIDER_NAME == "anthropic" else None,
)
DEFAULT_PROVIDER = build_batch_provider(
    DEFAULT_PROVIDER_NAME,
    model=DEFAULT_PROVIDER_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    verbosity=DEFAULT_VERBOSITY,
    gemini_thinking_budget=resolve_gemini_thinking_budget(),
    gemini_include_thoughts=resolve_gemini_include_thoughts(),
    anthropic_thinking_budget=DEFAULT_ANTHROPIC_THINKING_BUDGET,
    anthropic_max_tokens=resolve_anthropic_max_tokens(
        thinking_budget=DEFAULT_ANTHROPIC_THINKING_BUDGET,
    ),
)


def default_requests_folder(provider_name: str) -> Path:
    return Path("../requests") / normalize_provider_name(provider_name)


def default_responses_folder(provider_name: str) -> Path:
    return Path("../responses") / normalize_provider_name(provider_name)


def load_file_with_includes(filepath: Path) -> Generator[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("@include"):
                included_file = line.split()[1]
                included_file_path = filepath.parent / included_file
                yield from load_file_with_includes(included_file_path)
            elif line.startswith("@list_files"):
                list_dir = line.split()[1]
                list_dir_path = filepath.parent / list_dir

                prefix = line.split()[2] if len(line.split()) > 2 else list_dir
                filter = line.split()[3] if len(line.split()) > 3 else "*"

                def yield_from_directory(list_dir_path: Path, prefix: str) -> Generator[str]:
                    for item in sorted(list_dir_path.iterdir()):
                        if item.is_file():
                            if not item.match(filter):
                                continue
                            yield f"--- START OF FILE: {prefix}/{item.name} ---\n"
                            yield from load_file_with_includes(item)
                            yield f"--- END OF FILE: {prefix}/{item.name} ---\n"
                        elif item.is_dir():
                            yield from yield_from_directory(item, prefix + "/" + item.name)

                yield from yield_from_directory(list_dir_path, prefix)
            else:
                yield line


def load_messages(system_filename: Path, user_filename: Path):
    system_message = "".join(load_file_with_includes(system_filename))
    user_message = "".join(load_file_with_includes(user_filename))

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def extract_code_block(response_text: str) -> str | None:
    # This regex looks for content between triple backticks, possibly with a language specifier.
    pattern = r"```(?:\w*\+*\n)?(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def create_request(system_filename: Path, user_filename: Path, provider: BatchProvider = DEFAULT_PROVIDER) -> dict[str, Any]:
    messages = load_messages(system_filename, user_filename)
    return provider.build_request(messages)


def prepare_batch_line(system_filename: Path, user_filename: Path, repetition: int, provider: BatchProvider = DEFAULT_PROVIDER) -> dict[str, Any]:
    name = user_filename.stem
    if name.endswith(".prompt"):
        name = name[:-7]
    custom_id = provider.format_custom_id(name, repetition)
    body = create_request(system_filename, user_filename, provider=provider)
    return provider.create_batch_entry(custom_id, body)


def save_code_block(response_text: str, output_folder: Path):
    output_file = output_folder / "code.cpp"

    code_block = extract_code_block(response_text)
    if code_block is not None:
        logger.info("Saving extracted code block.")
    else:
        logger.info("No code block found in response. Saving full response instead.")
        code_block = response_text

    output_file.write_text(code_block, encoding="utf-8")


def save_error_record(error_payload: dict[str, Any], output_folder: Path) -> None:
    error_file = output_folder / "error.json"
    error_file.write_text(json.dumps(error_payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_batch_file(
    system_filename: Path,
    user_filenames: List[Path],
    output_filename: Path,
    repetitions: int = 5,
    start: int = 1,
    provider: BatchProvider = DEFAULT_PROVIDER,
):
    if system_filename.exists() is False:
        raise FileNotFoundError(f"System prompt file not found: {system_filename}")

    with open(output_filename, "w", encoding="utf-8") as f:
        for prompt in user_filenames:
            for repetition in range(start, start + repetitions):
                line = prepare_batch_line(system_filename, prompt, repetition, provider=provider)
                json.dump(line, f)
                f.write("\n")

    logger.info(f"Prepared batch file: {output_filename}")


def parse_batch_output(output_filename: Path, output_folder: Path, provider: BatchProvider = DEFAULT_PROVIDER):
    with open(output_filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                response = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d in %s: %s", line_number, output_filename, exc)
                continue

            try:
                parsed = provider.parse_batch_result(response)
            except Exception as exc:
                logger.warning("Failed to parse line %d in %s: %s", line_number, output_filename, exc)
                continue

            current_output_folder = output_folder / parsed.custom_id
            current_output_folder.mkdir(exist_ok=True, parents=True)

            if parsed.error_payload is not None:
                logger.warning("Saving error payload for %s from %s.", parsed.custom_id, output_filename)
                save_error_record(parsed.error_payload, current_output_folder)
                continue

            try:
                response_text = provider.extract_answer(parsed.response_body)
                save_code_block(response_text, current_output_folder)
                provider.estimate_cost(parsed.response_body, current_output_folder)
            except Exception as exc:
                logger.warning("Failed to extract answer for %s from %s: %s", parsed.custom_id, output_filename, exc)
                save_error_record({"message": str(exc), "raw": response}, current_output_folder)

    logger.info(f"Parsed batch output: {output_filename}")


def collect_benchmark_sources(source_folder: Path, extension: str) -> dict[str, List[Path]]:
    benchmarks: dict[str, List[Path]] = {
        "linear-algebra": sorted((source_folder / "linear-algebra").rglob(f"*{extension}")),
        "datamining": sorted((source_folder / "datamining").rglob(f"*{extension}")),
        "stencil": sorted((source_folder / "stencils").rglob(f"*{extension}")),
        "medley": sorted((source_folder / "medley").rglob(f"*{extension}")),
    }

    normalized: dict[str, List[Path]] = {}
    for category, files in benchmarks.items():
        non_empty_files = [
            path
            for path in files
            if path.exists() and path.is_file() and path.stat().st_size > 0
        ]
        non_empty_files.sort()
        normalized[category] = non_empty_files

    return normalized


def prepare_task(
    task: str,
    prompts_folder: Path,
    requests_folder: Path,
    source_folder: Path,
    extension: str,
    framework: str,
    repetitions: int = DEFAULT_REPETITIONS,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    category_files = collect_benchmark_sources(source_folder, extension)

    linear_algebra_files = category_files["linear-algebra"]
    datamining_files = category_files["datamining"]
    stencil_files = category_files["stencil"]
    medley_files = category_files["medley"]

    logger.info(f"Found {len(linear_algebra_files)} linear-algebra files.")
    logger.info(f"Found {len(datamining_files)} datamining files.")
    logger.info(f"Found {len(stencil_files)} stencil files.")
    logger.info(f"Found {len(medley_files)} medley files.")

    system_prompt = prompts_folder / f"{framework}.prompt.md"

    requests_folder.mkdir(exist_ok=True, parents=True)

    file_sets = [
        linear_algebra_files,
        datamining_files,
        stencil_files,
        medley_files
    ]

    source_files = sorted({path for files in category_files.values() for path in files})
    test_files = [file for file in source_files if file.stem in TEST_BENCHMARKS]

    file_sets.append(source_files)
    file_sets.append(test_files)

    for category, files in zip(CATEGORIES, file_sets):
        try:
            prepare_batch_file(
                system_prompt,
                files,
                requests_folder / f"{framework}_{category}.jsonl",
                repetitions=repetitions,
                provider=provider,
            )
        except Exception as e:
            logger.error(f"Error preparing {framework} {task} {category} requests: {e}")

    # do partial fulls:
    try:
        with open(requests_folder / f"{framework}_full.jsonl", "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        for i in range(0, len(all_lines), 5):
            batch_lines = all_lines[i : i + 5]
            with open(requests_folder / f"{framework}_full_part{(i // 5) + 1:02d}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(batch_lines)
    except Exception as e:
        logger.error(f"Error preparing {framework} {task} full partial requests: {e}")


def process_task(
    task: str,
    responses_folder: Path,
    output_folder: Path,
    framework: str,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    responses_folder.mkdir(exist_ok=True, parents=True)

    for category in CATEGORIES:
        output = output_folder / framework / category
        response = responses_folder / f"{framework}_{category}.jsonl"

        try:
            output.mkdir(exist_ok=True, parents=True)
            parse_batch_output(response, output, provider=provider)
        except Exception as e:
            logger.error(f"Error parsing {framework} {task} {category} responses: {e}")


    # process partial fulls
    full_output = output_folder / framework / "full"
    full_output.mkdir(exist_ok=True, parents=True)
    part_index = 1
    while True:
        response = responses_folder / f"{framework}_full_part{part_index:02d}.jsonl"
        if not response.exists():
            break
        try:
            parse_batch_output(response, full_output, provider=provider)
        except Exception as e:
            logger.error(f"Error parsing {framework} {task} full part {part_index:02d} responses: {e}")
        part_index += 1


def prepare_translation(
    prompts_folder: Path,
    requests_folder: Path,
    source_folder: Path,
    extension: str,
    framework: str,
    repetitions: int = DEFAULT_REPETITIONS,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    return prepare_task("translation", prompts_folder, requests_folder, source_folder, extension, framework, repetitions=repetitions, provider=provider)


def process_translation(responses_folder: Path, output_folder: Path, framework: str, provider: BatchProvider = DEFAULT_PROVIDER) -> None:
    return process_task("translation", responses_folder, output_folder, framework, provider=provider)


def prepare_optimization(
    prompts_folder: Path,
    requests_folder: Path,
    source_folder: Path,
    extension: str,
    framework: str,
    repetitions: int = DEFAULT_REPETITIONS,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    return prepare_task("optimization", prompts_folder, requests_folder, source_folder, extension, framework, repetitions=repetitions, provider=provider)


def process_optimization(responses_folder: Path, output_folder: Path, framework: str, provider: BatchProvider = DEFAULT_PROVIDER) -> None:
    return process_task("optimization", responses_folder, output_folder, framework, provider=provider)


def prepare_to_model(
    prompts_folder: Path,
    requests_folder: Path,
    source_folder: Path,
    extension: str,
    framework: str,
    repetitions: int = DEFAULT_REPETITIONS,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    return prepare_task("to_model", prompts_folder, requests_folder, source_folder, extension, framework, repetitions=repetitions, provider=provider)


def process_to_model(responses_folder: Path, output_folder: Path, framework: str, provider: BatchProvider = DEFAULT_PROVIDER) -> None:
    return process_task("to_model", responses_folder, output_folder, framework, provider=provider)


def prepare_from_model(
    prompts_folder: Path,
    requests_folder: Path,
    source_folder: Path,
    extension: str,
    framework: str,
    to_model_root: Path | None = None,
    repetitions: int = DEFAULT_REPETITIONS,
    provider: BatchProvider = DEFAULT_PROVIDER,
) -> None:
    category_files = collect_benchmark_sources(source_folder, extension)

    system_prompt = prompts_folder / f"{framework}.prompt.md"
    if system_prompt.exists() is False:
        raise FileNotFoundError(f"System prompt file not found: {system_prompt}")

    if to_model_root is None:
        to_model_root = Path("../to_model")

    model_framework_dir = to_model_root / framework
    if model_framework_dir.exists() is False:
        logger.warning(f"Model folder not found for framework '{framework}': {model_framework_dir}")
        return

    requests_folder.mkdir(exist_ok=True, parents=True)

    logger.info(f"Preparing from_model requests for framework '{framework}'.")
    for category, files in category_files.items():
        logger.info(f"Found {len(files)} {category} source files.")

    source_by_base: dict[str, Path] = {}
    category_bases: dict[str, List[str]] = {}
    for category, files in category_files.items():
        bases: list[str] = []
        for path in files:
            base = path.stem
            source_by_base[base] = path
            bases.append(base)
        category_bases[category] = sorted(bases)

    all_bases = sorted(source_by_base.keys())
    category_bases["full"] = all_bases
    category_bases["test"] = [base for base in all_bases if base in TEST_BENCHMARKS]

    models_by_base: dict[str, dict[str, Path]] = {}
    for category_dir in sorted([d for d in model_framework_dir.iterdir() if d.is_dir()]):
        for custom_dir in sorted([d for d in category_dir.iterdir() if d.is_dir()]):
            custom_id = custom_dir.name
            base = custom_id.rsplit("-", 1)[0] if "-" in custom_id else custom_id

            model_file = custom_dir / "code.cpp"
            if model_file.exists() is False:
                candidates = [f for f in custom_dir.iterdir() if f.is_file() and f.name != "costs.json"]
                candidates.sort()
                if not candidates:
                    logger.warning(f"No model artifact found in {custom_dir}.")
                    continue
                model_file = candidates[0]

            base_models = models_by_base.setdefault(base, {})
            base_models[custom_id] = model_file

    if not models_by_base:
        logger.warning(f"No models found for framework '{framework}' in {model_framework_dir}.")

    system_content = "".join(load_file_with_includes(system_prompt))

    def build_user_message(source_path: Path, model_entries: list[tuple[str, Path]]) -> str:
        parts: list[str] = []
        source_content = "".join(load_file_with_includes(source_path))
        parts.append(f"--- START OF SOURCE: {source_path.name} ---\n{source_content}\n--- END OF SOURCE: {source_path.name} ---")

        for custom_id, model_path in model_entries:
            model_content = "".join(load_file_with_includes(model_path))
            parts.append(
                f"--- START OF MODEL: {custom_id} ---\n{model_content}\n--- END OF MODEL: {custom_id} ---"
            )

        return "\n\n".join(parts)

    for category, base_list in category_bases.items():
        try:
            written = 0
            output_filename = requests_folder / f"{framework}_{category}.jsonl"
            with open(output_filename, "w", encoding="utf-8") as handle:
                for base in base_list:
                    source_path = source_by_base.get(base)
                    if source_path is None:
                        logger.warning(f"No source path found for base '{base}' in category '{category}'.")
                        continue

                    model_dict = models_by_base.get(base)
                    if not model_dict:
                        logger.warning(f"No model available for source '{base}' (framework '{framework}'). Skipping.")
                        continue

                    sorted_models = sorted(model_dict.items())

                    user_content = build_user_message(source_path, sorted_models)

                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ]
                    request_body = provider.build_request(messages)

                    for repetition in range(1, repetitions + 1):
                        custom_id = provider.format_custom_id(base, repetition)
                        request_line = provider.create_batch_entry(custom_id, request_body)
                        json.dump(request_line, handle)
                        handle.write("\n")
                        written += 1

            logger.info(f"Prepared from_model batch file: {output_filename} ({written} requests).")
        except Exception as e:
            logger.error(f"Error preparing {framework} from_model {category} requests: {e}")

    # do partial fulls:
    try:
        with open(requests_folder / f"{framework}_full.jsonl", "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        for i in range(0, len(all_lines), 5):
            batch_lines = all_lines[i : i + 5]
            with open(requests_folder / f"{framework}_full_part{(i // 5) + 1:02d}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(batch_lines)
    except Exception as e:
        logger.error(f"Error preparing {framework} from_model full partial requests: {e}")



def process_from_model(responses_folder: Path, output_folder: Path, framework: str, provider: BatchProvider = DEFAULT_PROVIDER) -> None:
    return process_task("from_model", responses_folder, output_folder, framework, provider=provider)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tool to prepare and process GPT querying batches.")
    parser.add_argument("--prompts-folder", type=Path, default=Path("../prompts"), help="Path to the prompts folder.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER_NAME, help="Provider label used for request and response folders. Aliases: google->gemini, claude/opus->anthropic.")
    parser.add_argument("--model", default=None, help="Model identifier written into generated batch requests. Defaults to provider-specific GPT_QUERYING_*_MODEL, then GPT_QUERYING_MODEL.")
    parser.add_argument(
        "--reasoning-effort",
        "--thinking-effort",
        dest="reasoning_effort",
        default=None,
        help="Generic reasoning effort. Used directly by OpenAI, passed through to Gemini with xhigh downgraded to high, and mapped to Anthropic effort or legacy thinking budgets depending on the selected model.",
    )
    parser.add_argument(
        "--verbosity",
        default=None,
        help="Verbosity written into generated batch requests. Currently applies to OpenAI requests only.",
    )
    parser.add_argument("--gemini-thinking-budget", type=int, default=None, help="Explicit Gemini thinking budget. If set, Gemini requests do not also send generic reasoning controls.")
    parser.add_argument(
        "--gemini-include-thoughts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether Gemini should include thought summaries in the response.",
    )
    parser.add_argument("--anthropic-thinking-budget", type=int, default=None, help="Explicit Anthropic extended thinking budget.")
    parser.add_argument("--anthropic-max-tokens", type=int, default=None, help="Anthropic max_tokens value. Defaults to max(300000, thinking_budget + 4096) with the 300k batch beta enabled.")
    parser.add_argument("--requests-folder", type=Path, help="Path to the requests folder. Defaults to ../requests/<provider>.")
    parser.add_argument("--responses-folder", type=Path, help="Path to the responses folder. Defaults to ../responses/<provider>.")

    parser.add_argument("--generated-folder", type=Path, default=Path("../generated"), help="Path to the generated folder.")
    parser.add_argument("--optimization-folder", type=Path, default=Path("../optimization"), help="Path to the optimization folder.")
    parser.add_argument("--to-model-folder", type=Path, default=Path("../to_model"), help="Path to the to_model folder.")
    parser.add_argument("--from-model-folder", type=Path, default=Path("../from_model"), help="Path to the from_model folder.")

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Number of repetitions per prompt. Defaults to GPT_QUERYING_REPETITIONS or GPT_QUERYING_REPS.",
    )

    parser.add_argument("--parse", action="store_true", default=False, help="Do a parse step (otherwise, do a generate phase)")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    provider_name = normalize_provider_name(args.provider)
    reasoning_effort = resolve_reasoning_effort(args.reasoning_effort)
    verbosity = resolve_verbosity(args.verbosity)
    model = resolve_provider_model(provider_name, args.model)
    gemini_thinking_budget = resolve_gemini_thinking_budget(args.gemini_thinking_budget)
    gemini_include_thoughts = resolve_gemini_include_thoughts(args.gemini_include_thoughts)
    anthropic_thinking_budget = resolve_anthropic_thinking_budget(
        args.anthropic_thinking_budget,
        reasoning_effort=reasoning_effort,
        model=model if provider_name == "anthropic" else None,
    )
    anthropic_max_tokens = resolve_anthropic_max_tokens(
        args.anthropic_max_tokens,
        thinking_budget=anthropic_thinking_budget,
    )

    # Folder configuration
    prompts_folder = args.prompts_folder
    requests_folder = args.requests_folder or default_requests_folder(provider_name)
    responses_folder = args.responses_folder or default_responses_folder(provider_name)
    provider = build_batch_provider(
        provider_name,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        gemini_thinking_budget=gemini_thinking_budget,
        gemini_include_thoughts=gemini_include_thoughts,
        anthropic_thinking_budget=anthropic_thinking_budget,
        anthropic_max_tokens=anthropic_max_tokens,
    )

    c_source_folder = Path("../PolybenchC-4.2.1")
    noarr_source_folder = Path("../PolybenchC-Noarr")
    halide_source_folder = Path("../PolybenchC-Halide")
    exo_source_folder = Path("../PolybenchC-Exo")

    frameworks = [
        ("c", c_source_folder, ".c"),
        ("exo", exo_source_folder, ".c"),
        ("noarr", noarr_source_folder, ".cpp"),
        ("halide", halide_source_folder, ".cpp"),
    ]

    generated_folder = args.generated_folder
    for framework, _, _ in frameworks:
        if args.parse:
            process_translation(responses_folder / "translation", generated_folder, framework=framework, provider=provider)
        else:
            prepare_translation(prompts_folder / "translation", requests_folder / "translation", c_source_folder, extension=".c", framework=framework, repetitions=args.repetitions, provider=provider)

    optimization_variants = [
        "naive",
        "cache",
        "arithmetic",
        "parallelism",
        "structure",
        "all_hints",
        "choose_hints",
    ]

    optimization_folder = args.optimization_folder
    for variant in optimization_variants:
        for framework, source_folder, extension in frameworks:
            if args.parse:
                process_optimization(responses_folder / "optimization", optimization_folder, framework=f"{framework}_{variant}", provider=provider)
            else:
                prepare_optimization(prompts_folder / "optimization", requests_folder / "optimization", source_folder, extension=extension, framework=f"{framework}_{variant}", repetitions=args.repetitions, provider=provider)

    to_model_folder = args.to_model_folder
    for framework, source_folder, extension in frameworks:
        if args.parse:
            process_to_model(responses_folder / "to_model", to_model_folder, framework=framework, provider=provider)
        else:
            prepare_to_model(prompts_folder / "to_model", requests_folder / "to_model", source_folder, extension=extension, framework=framework, repetitions=args.repetitions, provider=provider)

    from_model_folder = args.from_model_folder
    for framework, source_folder, extension in frameworks:
        if args.parse:
            process_from_model(responses_folder / "from_model", from_model_folder, framework=framework, provider=provider)
        else:
            prepare_from_model(prompts_folder / "from_model", requests_folder / "from_model", source_folder, extension=extension, framework=framework, to_model_root=to_model_folder, repetitions=args.repetitions, provider=provider)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
