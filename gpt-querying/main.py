#!/usr/bin/env python3

from __future__ import annotations

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

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


DEFAULT_REPETITIONS = 5
TEST_BENCHMARKS = ("2mm", "floyd-warshall", "heat-3d", "gemm")
CATEGORIES = ["linear-algebra", "datamining", "stencil", "medley", "full", "test"]


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

    def get_custom_id(self, batch_result: dict[str, Any]) -> str:
        return batch_result["custom_id"]

    def extract_response_body(self, batch_result: dict[str, Any]) -> Any:
        return batch_result["response"]["body"]

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        return None

    @abstractmethod
    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        ...

    @abstractmethod
    def extract_answer(self, response: Any) -> str:
        ...


@dataclass
class OpenAIChatBatchProvider(BatchProvider):
    model: str
    reasoning_effort: str = "high"
    verbosity: str = "medium"
    cost_input: float = 0.625 / 1_000_000  # $0.625 per 1M tokens (Batch pricing)
    cost_cached_input: float = 0.0625 / 1_000_000  # $0.0625 per 1M tokens (Batch pricing; 10x cheaper for cached)
    cost_output: float = 5 / 1_000_000  # $5 per 1M tokens (Batch pricing)
    url: str = "/v1/chat/completions"
    method: str = "POST"

    def build_request(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
            "messages": messages,
        }

    def extract_response_body(self, batch_result: dict[str, Any]) -> Any:
        body = super().extract_response_body(batch_result)
        return munchify(body)

    def extract_answer(self, response: Any) -> str:
        return response.choices[0].message.content

    def estimate_cost(self, response: Any, output_folder: Path, time_taken: float | None = None) -> None:
        try:
            prompt = response.usage.prompt_tokens
            prompt_cached = response.usage.prompt_tokens_details.cached_tokens
            completion = response.usage.completion_tokens
            reasoning = response.usage.completion_tokens_details.reasoning_tokens
        except AttributeError:
            logger.debug("OpenAI response missing usage details; skipping cost estimation.")
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


DEFAULT_PROVIDER = OpenAIChatBatchProvider(model="gpt-5.1-2025-11-13")


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
        for line in f:
            response = json.loads(line)
            custom_id = provider.get_custom_id(response)
            response_body = provider.extract_response_body(response)

            current_output_folder = output_folder / custom_id
            current_output_folder.mkdir(exist_ok=True, parents=True)

            response_text = provider.extract_answer(response_body)
            save_code_block(response_text, current_output_folder)
            provider.estimate_cost(response_body, current_output_folder)

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
    parser.add_argument("--requests-folder", type=Path, default=Path("../requests") / "openai", help="Path to the requests folder.")
    parser.add_argument("--responses-folder", type=Path, default=Path("../responses") / "openai", help="Path to the responses folder.")

    parser.add_argument("--generated-folder", type=Path, default=Path("../generated"), help="Path to the generated folder.")
    parser.add_argument("--optimization-folder", type=Path, default=Path("../optimization"), help="Path to the optimization folder.")
    parser.add_argument("--to-model-folder", type=Path, default=Path("../to_model"), help="Path to the to_model folder.")
    parser.add_argument("--from-model-folder", type=Path, default=Path("../from_model"), help="Path to the from_model folder.")

    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Number of repetitions per prompt.")

    parser.add_argument("--parse", action="store_true", default=False, help="Do a parse step (otherwise, do a generate phase)")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Folder configuration
    prompts_folder = args.prompts_folder
    requests_folder = args.requests_folder
    responses_folder = args.responses_folder

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
            process_translation(responses_folder / "translation", generated_folder, framework=framework)
        else:
            prepare_translation(prompts_folder / "translation", requests_folder / "translation", c_source_folder, extension=".c", framework=framework, repetitions=args.repetitions)

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
                process_optimization(responses_folder / "optimization", optimization_folder, framework=f"{framework}_{variant}")
            else:
                prepare_optimization(prompts_folder / "optimization", requests_folder / "optimization", source_folder, extension=extension, framework=f"{framework}_{variant}", repetitions=args.repetitions)

    to_model_folder = args.to_model_folder
    for framework, source_folder, extension in frameworks:
        if args.parse:
            process_to_model(responses_folder / "to_model", to_model_folder, framework=framework)
        else:
            prepare_to_model(prompts_folder / "to_model", requests_folder / "to_model", source_folder, extension=extension, framework=framework, repetitions=args.repetitions)

    from_model_folder = args.from_model_folder
    for framework, source_folder, extension in frameworks:
        if args.parse:
            process_from_model(responses_folder / "from_model", from_model_folder, framework=framework)
        else:
            prepare_from_model(prompts_folder / "from_model", requests_folder / "from_model", source_folder, extension=extension, framework=framework, to_model_root=to_model_folder, repetitions=args.repetitions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
