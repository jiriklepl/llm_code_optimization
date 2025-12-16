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
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass
class Candidate:
    """Holds aggregated data for a single framework/algorithm repetition."""

    framework: str
    version: str
    algorithm: str
    repetition: int
    validity_score: int
    total_time: float
    mean_times: Dict[str, Optional[float]]


@dataclass(unsafe_hash=True)
class TimeReportKey:
    """Key for time report entries."""

    framework: str
    version: str
    algorithm: str
    repetition: int
    dataset_size: str


@dataclass(unsafe_hash=True)
class ValidationReportKey:
    """Key for validation report entries."""

    framework: str
    version: str
    algorithm: str
    repetition: int


@dataclass(unsafe_hash=True)
class BestCandidateKey:
    """Key for best candidate entries."""

    framework: str
    version: str
    algorithm: str


@dataclass(unsafe_hash=True)
class UnframeworkedBestKey:
    """Key for unframeworked best entries."""

    version: str
    algorithm: str


@dataclass(unsafe_hash=True)
class TotalBestKey:
    """Key for total best entries."""

    algorithm: str


VALID_STATUS_TRUE = {"valid", "true"}
VALID_STATUS_FALSE = {"invalid", "illegal", "timeout", "false"}


def is_valid_status(value: str | None) -> bool:
    """Normalize status strings into a boolean flag."""

    status = (value or "").strip().lower()
    if status in VALID_STATUS_TRUE:
        return True
    if status in VALID_STATUS_FALSE:
        return False
    return False


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
) -> Tuple[Dict[TimeReportKey, float], Set[str]]:
    """Load time reports and compute mean time per (framework, algorithm, repetition, dataset_size)."""

    grouped: Dict[TimeReportKey, List[float]] = defaultdict(list)
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
                    version = row["version"].strip()
                    algorithm = row["algorithm"].strip()
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
                    version=version,
                    algorithm=algorithm,
                    repetition=repetition,
                    dataset_size=dataset_size,
                )
                grouped[key].append(time_value)

    mean_times = {key: sum(values) / len(values) for key, values in grouped.items()}
    return mean_times, dataset_sizes


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
                    version = row["version"].strip()
                    algorithm = row["algorithm"].strip()
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

                is_valid = is_valid_status(row.get("valid"))
                key = ValidationReportKey(
                    framework=framework,
                    version=version,
                    algorithm=algorithm,
                    repetition=repetition,
                )
                validity_scores[key] += 1 if is_valid else 0

    return validity_scores, dataset_sizes


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
        if candidate.total_time < current.total_time:
            return True

        if (
            candidate.total_time == current.total_time
            and candidate.repetition < current.repetition
        ):
            return True

    return False


def select_best_candidates(
    dataset_sizes: Sequence[str],
    mean_times: Mapping[TimeReportKey, float],
    validity_scores: Mapping[ValidationReportKey, int],
) -> Dict[BestCandidateKey, Candidate]:
    """Pick the best repetition for every (framework, algorithm) pair."""

    best: Dict[BestCandidateKey, Candidate] = {}
    tracked_sizes = list(dataset_sizes)
    for validation_key, validity_score in validity_scores.items():
        framework, version, algorithm, repetition = (
            validation_key.framework,
            validation_key.version,
            validation_key.algorithm,
            validation_key.repetition,
        )

        mean_by_size = {
            size: mean_times.get(
                TimeReportKey(
                    framework=framework,
                    version=version,
                    algorithm=algorithm,
                    repetition=repetition,
                    dataset_size=size,
                )
            )
            for size in tracked_sizes
        }

        total_time = sum(
            time
            for time in mean_by_size.values()
            if time is not None and math.isfinite(time)
        )

        candidate = Candidate(
            framework=framework,
            version=version,
            algorithm=algorithm,
            repetition=repetition,
            validity_score=validity_score,
            total_time=total_time,
            mean_times=mean_by_size,
        )

        key = BestCandidateKey(
            framework=framework,
            version=version,
            algorithm=algorithm,
        )

        current = best.get(key)
        if is_better_candidate(candidate, current):
            best[key] = candidate

    return best


def select_unframeworked_bests(
    candidates: Mapping[BestCandidateKey, Candidate],
) -> Dict[UnframeworkedBestKey, Candidate]:
    """Select the best framework per (version, algorithm) pair."""

    best: Dict[UnframeworkedBestKey, Candidate] = {}

    for key, candidate in candidates.items():
        version, algorithm = key.version, key.algorithm
        unframeworked_key = UnframeworkedBestKey(version=version, algorithm=algorithm)
        current = best.get(unframeworked_key)
        if is_better_candidate(candidate, current):
            best[unframeworked_key] = candidate

    return best


def select_total_bests(
    candidates: Mapping[UnframeworkedBestKey, Candidate],
) -> Dict[TotalBestKey, Candidate]:
    """Select the best version per algorithm."""

    best: Dict[TotalBestKey, Candidate] = {}

    for key, candidate in candidates.items():
        algorithm = key.algorithm
        total_best_key = TotalBestKey(algorithm=algorithm)
        current = best.get(total_best_key)
        if is_better_candidate(candidate, current):
            best[total_best_key] = candidate

    return best


def format_total_time(value: float) -> str:
    """Format total time or return an empty string for non-finite values."""

    if not math.isfinite(value):
        return ""

    return f"{value:.6f}"


def emit_best_candidates(
    dataset_sizes: Sequence[str], candidates: Mapping[BestCandidateKey, Candidate]
) -> None:
    """Print the best repetition per framework/algorithm."""

    if not candidates:
        print("No non-candidates found.")
        return

    print("# Best repetitions per framework, version, and algorithm")
    header = [
        "framework",
        "version",
        "algorithm",
        "repetition",
        "validity_score",
        "total_time",
    ] + [f"time_{size}" for size in dataset_sizes]
    print(",".join(header))

    for key in sorted(
        candidates.keys(), key=lambda item: (item.algorithm, item.framework)
    ):
        candidate = candidates[key]
        total_time = format_total_time(candidate.total_time)
        row = [
            candidate.framework,
            candidate.version,
            candidate.algorithm,
            str(candidate.repetition),
            str(candidate.validity_score),
            total_time,
        ]

        for dataset_size in dataset_sizes:
            row.append(format_float(candidate.mean_times.get(dataset_size)))

        print(",".join(row))


def emit_unframeworked_bests(
    dataset_sizes: Sequence[str], candidates: Mapping[UnframeworkedBestKey, Candidate]
) -> None:
    """Print the best framework per version/algorithm."""

    if not candidates:
        print("No unframeworked candidates found.")
        return

    print("# Best frameworks per version and algorithm")
    header = [
        "version",
        "algorithm",
        "framework",
        "repetition",
        "validity_score",
        "total_time",
    ] + [f"time_{size}" for size in dataset_sizes]
    print(",".join(header))

    for key in sorted(
        candidates.keys(), key=lambda item: (item.algorithm, item.version)
    ):
        candidate = candidates[key]
        total_time = format_total_time(candidate.total_time)
        row = [
            candidate.version,
            candidate.algorithm,
            candidate.framework,
            str(candidate.repetition),
            str(candidate.validity_score),
            total_time,
        ]

        for dataset_size in dataset_sizes:
            row.append(format_float(candidate.mean_times.get(dataset_size)))

        print(",".join(row))


def emit_total_bests(
    dataset_sizes: Sequence[str], candidates: Mapping[TotalBestKey, Candidate]
) -> None:
    """Print the best version per algorithm."""

    if not candidates:
        print("No total candidates found.")
        return

    print("# Best versions per algorithm")
    header = [
        "algorithm",
        "version",
        "framework",
        "repetition",
        "validity_score",
        "total_time",
    ] + [f"time_{size}" for size in dataset_sizes]
    print(",".join(header))

    for key in sorted(candidates.keys(), key=lambda item: item.algorithm):
        candidate = candidates[key]
        total_time = format_total_time(candidate.total_time)
        row = [
            candidate.algorithm,
            candidate.version,
            candidate.framework,
            str(candidate.repetition),
            str(candidate.validity_score),
            total_time,
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

    mean_times, time_sizes = load_time_reports(time_reports)
    validity_scores, validation_sizes = load_validation_reports(validation_reports)
    dataset_sizes = sorted(time_sizes | validation_sizes)

    candidates = select_best_candidates(dataset_sizes, mean_times, validity_scores)
    emit_best_candidates(dataset_sizes, candidates)

    unframeworked_candidates = select_unframeworked_bests(candidates)
    emit_unframeworked_bests(dataset_sizes, unframeworked_candidates)

    total_bests = select_total_bests(unframeworked_candidates)
    emit_total_bests(dataset_sizes, total_bests)


if __name__ == "__main__":
    main()
