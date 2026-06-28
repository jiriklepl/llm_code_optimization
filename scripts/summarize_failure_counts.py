#!/usr/bin/env python3
"""Print optimization failure-category paper numbers from failure_counts.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

COUNT_COLUMNS = [
    "attempt_count",
    "validated_attempt_count",
    "invalid_attempt_count",
    "compile_error_count",
    "runtime_error_count",
    "numerical_mismatch_count",
    "timeout_count",
    "unclassified_invalid_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "failure_counts_csv",
        nargs="?",
        default="results/plots/20260408/failure_counts.csv",
        type=Path,
        help=(
            "CSV produced by scripts/analyze_attempts.py "
            "(default: results/plots/20260408/failure_counts.csv)."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"framework", *COUNT_COLUMNS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def add_counts(target: dict[str, int], row: dict[str, str]) -> None:
    for column in COUNT_COLUMNS:
        target[column] += int(row[column])


def print_counts(prefix: str, counts: dict[str, int]) -> None:
    print(f"{prefix} total optimization attempts: {counts['attempt_count']}")
    print(f"{prefix} validated attempts: {counts['validated_attempt_count']}")
    print(f"{prefix} invalid attempts: {counts['invalid_attempt_count']}")
    print(f"{prefix} compile-error diagnostic count: {counts['compile_error_count']}")
    print(f"{prefix} runtime-error diagnostic count: {counts['runtime_error_count']}")
    print(
        f"{prefix} numerical-mismatch diagnostic count: "
        f"{counts['numerical_mismatch_count']}"
    )
    print(f"{prefix} timeout diagnostic count: {counts['timeout_count']}")
    print(
        f"{prefix} unclassified-invalid diagnostic count: "
        f"{counts['unclassified_invalid_count']}"
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(args.failure_counts_csv)

    total: dict[str, int] = defaultdict(int)
    by_framework: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        add_counts(total, row)
        add_counts(by_framework[row["framework"]], row)

    print("Optimization failure-category summary")
    print(f"Source: {args.failure_counts_csv}")
    print_counts("All frameworks", total)
    print("By framework:")
    for framework in sorted(by_framework):
        print_counts(f"  {framework}", by_framework[framework])
    print(
        "Note: diagnostic reason columns are tags, not a disjoint partition; "
        "one invalid attempt can contribute to more than one reason count across "
        "dataset sizes."
    )


if __name__ == "__main__":
    main()
