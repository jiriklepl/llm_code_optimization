#!/usr/bin/env python3
"""Select the best repetitions per framework/algorithm for a results directory."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_DATA_TYPE = "DOUBLE"


@dataclass
class Candidate:
    """Holds aggregated data for a single framework/algorithm repetition."""

    framework: str
    algorithm: str
    data_type: str
    repetition: int
    validity_score: int
    relative_gap: float
    mean_times: Dict[str, Optional[float]]


@dataclass(unsafe_hash=True)
class TimeReportKey:
    """Key for time report entries."""

    framework: str
    algorithm: str
    data_type: str
    repetition: int
    dataset_size: str


@dataclass(unsafe_hash=True)
class ValidationReportKey:
    """Key for validation report entries."""

    framework: str
    algorithm: str
    data_type: str
    repetition: int


@dataclass(unsafe_hash=True)
class BaselineKey:
    """Key for best candidate entries."""

    algorithm: str
    data_type: str
    dataset_size: str


@dataclass(unsafe_hash=True)
class BestCandidateKey:
    """Key for best candidate entries."""

    framework: str
    algorithm: str
    data_type: str


VALID_STATUS_TRUE = {"valid", "true"}
VALID_STATUS_FALSE = {
    "invalid",
    "illegal",
    "timeout",
    "false",
    "validity_error",
    "runtime_error",
    "compile_error",
}


def is_valid_status(value: str | None) -> bool:
    """Normalize status strings into a boolean flag."""

    status = (value or "").strip().lower()
    if status in VALID_STATUS_TRUE:
        return True
    if status in VALID_STATUS_FALSE:
        return False
    return False


def get_validation_status(row: Mapping[str, str]) -> str | None:
    """Return status from either the new status column or the legacy valid column."""

    return row.get("status") or row.get("valid")


def get_data_type(row: Mapping[str, str]) -> str:
    """Return the benchmark datatype, defaulting legacy reports to DOUBLE."""

    return (row.get("data_type") or DEFAULT_DATA_TYPE).strip() or DEFAULT_DATA_TYPE


def extract_dataset_size(path: Path) -> Optional[str]:
    """Return the dataset size suffix from files such as times_SMALL.csv."""

    stem = path.stem  # e.g., times_SMALL
    if "_" not in stem:
        return None

    return stem.split("_", 1)[1]


def format_float(value: Optional[float]) -> str:
    """Format a float to six decimals or return an empty string."""

    if value is None:
        return ""

    return f"{value:.6f}"


def load_time_reports(
    csv_paths: Sequence[Path],
) -> Tuple[Dict[TimeReportKey, float], Dict[BaselineKey, List[float]], Set[str]]:
    """Load time reports and compute mean time per (framework, algorithm, repetition, dataset_size)."""

    grouped: Dict[TimeReportKey, List[float]] = defaultdict(list)
    baseline_raw: Dict[BaselineKey, List[float]] = defaultdict(list)
    dataset_sizes: Set[str] = set()

    for csv_path in csv_paths:
        inferred_size = extract_dataset_size(csv_path)
        if inferred_size is not None:
            dataset_sizes.add(inferred_size)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                try:
                    framework = row["framework"].strip()
                    algorithm = row["algorithm"].strip()
                    data_type = get_data_type(row)
                except KeyError as exc:
                    raise KeyError(f"Missing column {exc} in {csv_path}") from exc

                dataset_size = row.get("dataset_size", "").strip() or inferred_size
                if dataset_size:
                    dataset_sizes.add(dataset_size)

                repetition_str = row.get("repetition", "").strip()
                time_str = row.get("time_s", "").strip()

                if not repetition_str or not time_str:
                    continue

                try:
                    repetition = int(repetition_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid repetition value {repetition_str!r} in {csv_path} line {line_number}"
                    ) from exc

                try:
                    time_value = float(time_str)
                except ValueError:
                    # Ignore non-numeric runtimes.
                    continue

                if not dataset_size:
                    continue

                key = TimeReportKey(
                    framework=framework,
                    algorithm=algorithm,
                    data_type=data_type,
                    repetition=repetition,
                    dataset_size=dataset_size,
                )
                grouped[key].append(time_value)
                if framework == "c":
                    baseline_raw[
                        BaselineKey(
                            algorithm=algorithm,
                            data_type=data_type,
                            dataset_size=dataset_size,
                        )
                    ].append(time_value)

    mean_times = {key: sum(values) / len(values) for key, values in grouped.items()}
    return mean_times, baseline_raw, dataset_sizes


def load_validation_reports(
    csv_paths: Sequence[Path],
) -> Tuple[Dict[ValidationReportKey, int], Set[str]]:
    """Load validation reports and sum validity counts per (framework, algorithm, repetition)."""

    validity_scores: Dict[ValidationReportKey, int] = defaultdict(int)
    dataset_sizes: Set[str] = set()

    for csv_path in csv_paths:
        inferred_size = extract_dataset_size(csv_path)
        if inferred_size is not None:
            dataset_sizes.add(inferred_size)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                try:
                    framework = row["framework"].strip()
                    algorithm = row["algorithm"].strip()
                    data_type = get_data_type(row)
                except KeyError as exc:
                    raise KeyError(f"Missing column {exc} in {csv_path}") from exc

                dataset_size = row.get("dataset_size", "").strip() or inferred_size
                if dataset_size:
                    dataset_sizes.add(dataset_size)

                repetition_str = row.get("repetition", "").strip()
                if not repetition_str:
                    continue

                try:
                    repetition = int(repetition_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid repetition value {repetition_str!r} in {csv_path} line {line_number}"
                    ) from exc

                is_valid = is_valid_status(get_validation_status(row))
                key = ValidationReportKey(
                    framework=framework,
                    algorithm=algorithm,
                    data_type=data_type,
                    repetition=repetition,
                )
                validity_scores[key] += 1 if is_valid else 0

    return validity_scores, dataset_sizes


def compute_relative_gap(
    mean_times: Mapping[TimeReportKey, float],
    baseline: Mapping[BaselineKey, float],
    framework: str,
    algorithm: str,
    data_type: str,
    repetition: int,
    dataset_sizes: Iterable[str],
) -> float:
    """Compute how closely the candidate matches the c baseline."""

    deltas: List[float] = []
    for dataset_size in dataset_sizes:
        baseline_time = baseline.get(
            BaselineKey(
                algorithm=algorithm,
                data_type=data_type,
                dataset_size=dataset_size,
            )
        )
        if baseline_time is None:
            continue

        candidate_time = mean_times.get(
            TimeReportKey(
                framework=framework,
                algorithm=algorithm,
                data_type=data_type,
                repetition=repetition,
                dataset_size=dataset_size,
            )
        )
        if candidate_time is None:
            return math.inf

        gap = abs(candidate_time - baseline_time)
        if baseline_time != 0:
            gap /= abs(baseline_time)
        deltas.append(gap)

    if not deltas:
        return math.inf

    return sum(deltas) / len(deltas)


def is_better_candidate(
    candidate: Candidate,
    current: Candidate | None = None,
) -> bool:
    """Decide whether candidate is better than current."""

    if current is None:
        return True

    if candidate.validity_score > current.validity_score:
        return True

    if candidate.validity_score == current.validity_score:
        if candidate.relative_gap < current.relative_gap:
            return True

        # Tie-breaker: lower repetition wins.
        if (
            candidate.relative_gap == current.relative_gap
            and candidate.repetition < current.repetition
        ):
            return True

    return False


def select_best_candidates(
    dataset_sizes: Sequence[str],
    mean_times: Mapping[TimeReportKey, float],
    baseline: Mapping[BaselineKey, float],
    validity_scores: Mapping[ValidationReportKey, int],
) -> Dict[BestCandidateKey, Candidate]:
    """Pick the best repetition for every (framework, algorithm) pair."""

    best: Dict[BestCandidateKey, Candidate] = {}
    tracked_sizes = list(dataset_sizes)
    for validation_key, validity_score in validity_scores.items():
        framework, algorithm, data_type, repetition = (
            validation_key.framework,
            validation_key.algorithm,
            validation_key.data_type,
            validation_key.repetition,
        )
        if framework == "c":
            continue

        relative_gap = compute_relative_gap(
            mean_times, baseline, framework, algorithm, data_type, repetition, tracked_sizes
        )
        mean_by_size = {
            size: mean_times.get(
                TimeReportKey(
                    framework=framework,
                    algorithm=algorithm,
                    data_type=data_type,
                    repetition=repetition,
                    dataset_size=size,
                )
            )
            for size in tracked_sizes
        }

        candidate = Candidate(
            framework=framework,
            algorithm=algorithm,
            data_type=data_type,
            repetition=repetition,
            validity_score=validity_score,
            relative_gap=relative_gap,
            mean_times=mean_by_size,
        )

        key = BestCandidateKey(
            framework=framework,
            algorithm=algorithm,
            data_type=data_type,
        )
        current = best.get(key)
        if is_better_candidate(candidate, current):
            best[key] = candidate

    return best


def emit_baseline_table(
    dataset_sizes: Sequence[str], baseline: Mapping[BaselineKey, float]
) -> None:
    """Print the averaged baseline runtimes per algorithm."""

    if not baseline:
        return

    print("# Baseline mean runtimes (framework=c)")
    header = ["algorithm", "data_type"] + [f"time_{size}" for size in dataset_sizes]
    print(",".join(header))

    for algorithm, data_type in sorted({(key.algorithm, key.data_type) for key in baseline.keys()}):
        row = [algorithm, data_type]
        for dataset_size in dataset_sizes:
            row.append(
                format_float(
                    baseline.get(
                        BaselineKey(
                            algorithm=algorithm,
                            data_type=data_type,
                            dataset_size=dataset_size,
                        )
                    )
                )
            )
        print(",".join(row))

    print()


def emit_best_candidates(
    dataset_sizes: Sequence[str], candidates: Mapping[BestCandidateKey, Candidate]
) -> None:
    """Print the best repetition per framework/algorithm."""

    if not candidates:
        print("No non-candidates found.")
        return

    print("# Best repetitions per framework and algorithm")
    header = [
        "framework",
        "algorithm",
        "data_type",
        "repetition",
        "validity_score",
        "relative_gap",
    ] + [f"time_{size}" for size in dataset_sizes]
    print(",".join(header))

    for key in sorted(
        candidates.keys(), key=lambda item: (item.algorithm, item.data_type, item.framework)
    ):
        candidate = candidates[key]
        relative_gap = (
            ""
            if not math.isfinite(candidate.relative_gap)
            else f"{candidate.relative_gap:.6f}"
        )
        row = [
            candidate.framework,
            candidate.algorithm,
            candidate.data_type,
            str(candidate.repetition),
            str(candidate.validity_score),
            relative_gap,
        ]

        for dataset_size in dataset_sizes:
            row.append(format_float(candidate.mean_times.get(dataset_size)))

        print(",".join(row))


def ensure_files_exist(files: Sequence[Path], file_type: str) -> None:
    """Ensure that the list of files is not empty."""

    if files:
        return

    print(f"No {file_type} files found.", file=sys.stderr)
    sys.exit(1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to a directory containing times_*.csv and validation_*.csv files.",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"{results_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    time_reports = sorted(results_dir.glob("times_*.csv"))
    validation_reports = sorted(results_dir.glob("validation_*.csv"))
    ensure_files_exist(time_reports, "time report")
    ensure_files_exist(validation_reports, "validation report")

    mean_times, baseline_samples, time_sizes = load_time_reports(time_reports)
    baseline_means = {
        key: sum(values) / len(values) for key, values in baseline_samples.items()
    }
    validity_scores, validation_sizes = load_validation_reports(validation_reports)
    dataset_sizes = sorted(time_sizes | validation_sizes)
    candidates = select_best_candidates(
        dataset_sizes, mean_times, baseline_means, validity_scores
    )

    emit_baseline_table(dataset_sizes, baseline_means)
    emit_best_candidates(dataset_sizes, candidates)


if __name__ == "__main__":
    main()
