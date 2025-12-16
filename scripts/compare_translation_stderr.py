#!/usr/bin/env python3

import math
import re
import sys
from typing import List


TOKEN_PATTERN = re.compile(r"\b(?:nan|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\b", re.IGNORECASE)


def extract_tokens(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    return TOKEN_PATTERN.findall(content)


def validate_tokens(tokens: List[str], label: str) -> List[float]:
    converted: List[float] = []
    for index, token in enumerate(tokens, start=1):
        try:
            converted.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"Token {index} in {label} is not a valid floating-point number: '{token}'"
            ) from exc
    return converted


def main() -> int:
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print(
            "Usage: compare_translation_stderr.py <baseline_file> <translated_file> <abs_epsilon> [<rel_epsilon>]",
            file=sys.stderr,
        )
        return 1

    baseline_path, translated_path, abs_epsilon_str = sys.argv[1:4]
    rel_epsilon_str = sys.argv[4] if len(sys.argv) == 5 else None

    try:
        abs_epsilon = float(abs_epsilon_str)
    except ValueError:
        print(f"Invalid absolute epsilon value: {abs_epsilon_str}", file=sys.stderr)
        return 1

    if abs_epsilon < 0:
        print(f"Absolute epsilon must be non-negative, got {abs_epsilon}", file=sys.stderr)
        return 1

    if rel_epsilon_str is not None:
        try:
            rel_epsilon = float(rel_epsilon_str)
        except ValueError:
            print(f"Invalid relative epsilon value: {rel_epsilon_str}", file=sys.stderr)
            return 1

        if rel_epsilon < 0:
            print(f"Relative epsilon must be non-negative, got {rel_epsilon}", file=sys.stderr)
            return 1
    else:
        rel_epsilon = 0.0

    try:
        baseline_tokens = extract_tokens(baseline_path)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        translated_tokens = extract_tokens(translated_path)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if len(baseline_tokens) != len(translated_tokens):
        print(
            f"Token count differs: baseline={len(baseline_tokens)} translated={len(translated_tokens)}",
            file=sys.stderr,
        )
        return 1

    try:
        baseline_values = validate_tokens(baseline_tokens, "baseline")
        translated_values = validate_tokens(translated_tokens, "translated")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    for index, (base_val, trans_val) in enumerate(
        zip(baseline_values, translated_values), start=1
    ):
        if math.isnan(base_val) and math.isnan(trans_val):
            continue
        if math.isnan(base_val) or math.isnan(trans_val) or abs(base_val - trans_val) > abs_epsilon + rel_epsilon * abs(base_val):
            print(
                f"Token {index} differs: {base_val} vs {trans_val} (tolerance {abs_epsilon} + {rel_epsilon} * |{base_val}| = {abs_epsilon + rel_epsilon * abs(base_val)})",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
