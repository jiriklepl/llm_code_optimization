#!/usr/bin/env python3

from __future__ import annotations

import os

PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "opus": "anthropic",
}

DEFAULT_PROVIDER_API_BASES: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1",
}

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "openai": "gpt-5.4-2026-03-05",
    "gemini": "gemini-3.1-pro-preview",
    "anthropic": "claude-opus-4-6",
}

ANTHROPIC_REASONING_BUDGETS: dict[str, int | None] = {
    "none": None,
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 49152,
}

ANTHROPIC_EFFORT_LEVELS: dict[str, str] = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None

    normalized = reasoning_effort.strip().lower()
    return normalized or None


def normalized_model_name(model: str | None) -> str:
    if model is None:
        return ""
    return model.strip().lower()


def is_gemini_3_model(model: str | None) -> bool:
    return normalized_model_name(model).startswith("gemini-3")


def is_gemini_3_flash_model(model: str | None) -> bool:
    normalized = normalized_model_name(model)
    return normalized.startswith("gemini-3") and "flash" in normalized


def is_gemini_2_5_pro_model(model: str | None) -> bool:
    return normalized_model_name(model).startswith("gemini-2.5-pro")


def resolve_gemini_thinking_level(model: str, reasoning_effort: str | None) -> str | None:
    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None or not is_gemini_3_model(model):
        return None

    if is_gemini_3_flash_model(model):
        gemini_3_flash_levels = {
            "none": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }
        return gemini_3_flash_levels.get(normalized)

    gemini_3_pro_levels = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "high",
    }
    return gemini_3_pro_levels.get(normalized)


def resolve_gemini_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None:
        return None
    if normalized == "xhigh":
        return "high"
    return normalized


def anthropic_supports_effort(model: str | None) -> bool:
    normalized = normalized_model_name(model)
    return normalized.startswith(
        (
            "claude-mythos",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        )
    )


def anthropic_prefers_adaptive_thinking(model: str | None) -> bool:
    normalized = normalized_model_name(model)
    return normalized.startswith(("claude-opus-4-6", "claude-sonnet-4-6"))


def env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue

        value = value.strip()
        if value:
            return value

    return default


def env_optional_str(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue

        value = value.strip()
        if value:
            return value

    return None


def env_int(*names: str, default: int, minimum: int = 1) -> int:
    value = env_optional_int(*names, minimum=minimum)
    return default if value is None else value


def env_optional_int(*names: str, minimum: int = 1) -> int | None:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue

        value = raw_value.strip()
        if not value:
            continue

        try:
            parsed = int(value)
        except ValueError as exc:
            joined_names = ", ".join(names)
            raise SystemExit(f"Expected an integer in {joined_names}, got {raw_value!r}.") from exc

        if parsed < minimum:
            joined_names = ", ".join(names)
            raise SystemExit(f"Expected {joined_names} to be >= {minimum}, got {parsed}.")

        return parsed

    return None


def env_optional_bool(*names: str) -> bool | None:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue

        value = raw_value.strip().lower()
        if not value:
            continue
        if value in truthy:
            return True
        if value in falsy:
            return False

        joined_names = ", ".join(names)
        raise SystemExit(f"Expected a boolean in {joined_names}, got {raw_value!r}.")

    return None


def normalize_provider_name(provider: str | None) -> str:
    if provider is None:
        provider = env_str("GPT_QUERYING_PROVIDER", default="openai")

    normalized = provider.strip().lower()
    if not normalized:
        return "openai"

    canonical = PROVIDER_ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(sorted(PROVIDER_ALIASES))
        raise SystemExit(f"Unsupported provider {provider!r}. Supported names and aliases: {supported}.")

    return canonical


def provider_env_prefix(provider: str) -> str:
    return normalize_provider_name(provider).upper().replace("-", "_")


DEFAULT_PROVIDER_NAME = normalize_provider_name(env_str("GPT_QUERYING_PROVIDER", default="openai"))
DEFAULT_REASONING_EFFORT = env_str(
    "GPT_QUERYING_REASONING_EFFORT",
    "GPT_QUERYING_THINKING_EFFORT",
    default="xhigh",
).strip().lower()
DEFAULT_VERBOSITY = env_str("GPT_QUERYING_VERBOSITY", default="medium")
DEFAULT_ANTHROPIC_VERSION = env_str(
    "GPT_QUERYING_ANTHROPIC_VERSION",
    "ANTHROPIC_VERSION",
    default="2023-06-01",
)
DEFAULT_ANTHROPIC_BETA_HEADER = env_str(
    "GPT_QUERYING_ANTHROPIC_BETA_HEADER",
    "ANTHROPIC_BETA_HEADER",
    default="output-300k-2026-03-24",
)
DEFAULT_ANTHROPIC_MAX_TOKENS = 300_000


def resolve_provider_model(provider: str, cli_model: str | None = None) -> str:
    provider_name = normalize_provider_name(provider)
    provider_prefix = provider_env_prefix(provider_name)

    return (
        (cli_model.strip() if cli_model and cli_model.strip() else None)
        or env_optional_str(f"GPT_QUERYING_{provider_prefix}_MODEL")
        or env_optional_str("GPT_QUERYING_MODEL", "GPT_QUERYING_MODEL_SNAPSHOT")
        or DEFAULT_PROVIDER_MODELS[provider_name]
    )


def resolve_reasoning_effort(cli_reasoning_effort: str | None = None) -> str:
    if cli_reasoning_effort is None:
        return DEFAULT_REASONING_EFFORT
    return cli_reasoning_effort.strip().lower()


def resolve_verbosity(cli_verbosity: str | None = None) -> str:
    if cli_verbosity is None:
        return DEFAULT_VERBOSITY
    return cli_verbosity.strip()


def resolve_gemini_thinking_budget(cli_budget: int | None = None) -> int | None:
    if cli_budget is not None:
        return cli_budget
    return env_optional_int("GPT_QUERYING_GEMINI_THINKING_BUDGET", minimum=1)


def resolve_gemini_include_thoughts(cli_include_thoughts: bool | None = None) -> bool:
    if cli_include_thoughts is not None:
        return cli_include_thoughts
    value = env_optional_bool("GPT_QUERYING_GEMINI_INCLUDE_THOUGHTS")
    return False if value is None else value


def anthropic_budget_for_reasoning_effort(reasoning_effort: str | None) -> int | None:
    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None:
        return ANTHROPIC_REASONING_BUDGETS[DEFAULT_REASONING_EFFORT]
    return ANTHROPIC_REASONING_BUDGETS.get(normalized)


def resolve_anthropic_effort(model: str, reasoning_effort: str | None = None) -> str | None:
    if not anthropic_supports_effort(model):
        return None

    normalized = normalize_reasoning_effort(reasoning_effort)
    if normalized is None:
        normalized = DEFAULT_REASONING_EFFORT

    return ANTHROPIC_EFFORT_LEVELS.get(normalized)


def resolve_anthropic_thinking_budget(
    cli_budget: int | None = None,
    reasoning_effort: str | None = None,
    model: str | None = None,
) -> int | None:
    if cli_budget is not None:
        return cli_budget

    env_budget = env_optional_int("GPT_QUERYING_ANTHROPIC_THINKING_BUDGET", minimum=1)
    if env_budget is not None:
        return env_budget

    if anthropic_supports_effort(model):
        return None

    return anthropic_budget_for_reasoning_effort(reasoning_effort)


def resolve_anthropic_max_tokens(
    cli_max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> int:
    configured = cli_max_tokens
    if configured is None:
        configured = env_optional_int("GPT_QUERYING_ANTHROPIC_MAX_TOKENS", minimum=1)

    base_default = DEFAULT_ANTHROPIC_MAX_TOKENS if configured is None else configured
    thinking_default = 0 if thinking_budget is None else thinking_budget + 4096
    return max(base_default, thinking_default)
