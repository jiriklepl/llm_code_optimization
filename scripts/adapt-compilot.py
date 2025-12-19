#!/usr/bin/env python3

"""Convert raw timing CSVs into the translation times/validation format.

The input is expected to have a header like:
    name;type;execution_times_ms
and `execution_times_ms` contains whitespace-separated measurements.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


TIMES_HEADER = [
    "framework",
    "version",
    "algorithm",
    "run",
    "time_s",
    "repetition",
    "dataset_size",
]

VALIDATION_HEADER = [
    "framework",
    "algorithm",
    "time_s",
    "speedup",
    "valid",
    "repetition",
    "dataset_size",
]

ALGORITHM_RENAMES = {
    "fdtd_2d": "fdtd-2d",
    "floyd_warshall": "floyd-warshall",
    "heat3d": "heat-3d",
    "jacobi1d": "jacobi-1d",
    "jacobi2d": "jacobi-2d",
    "seidel2d": "seidel-2d",
}


@dataclass
class Entry:
    algorithm: str
    dataset_size: str
    times_ms: List[float]

    @property
    def algorithm_normalized(self) -> str:
        if self.algorithm in ALGORITHM_RENAMES:
            return ALGORITHM_RENAMES[self.algorithm]
        return self.algorithm.replace("_", "-")

    @property
    def times_s(self) -> List[float]:
        return [ms / 1000.0 for ms in self.times_ms]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt raw timing CSV (name;type;execution_times_ms) into "
            "results/translation/<run_id>/times_<size>.csv and "
            "validation_<size>.csv"
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV file to convert.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/translation"),
        help="Base directory for generated files (default: results/translation).",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run identifier; defaults to a timestamp derived from the input filename.",
    )
    return parser.parse_args()


def derive_run_id(input_path: Path) -> str:
    """Build run id like 20251211_150643 from the input filename."""

    stem = input_path.stem  # e.g., 2025_12_11__15_06_43-MINI
    base = stem.split("-", maxsplit=1)[0]

    digits = re.sub(r"\D", "", base)
    if len(digits) >= 14:
        return f"{digits[:8]}_{digits[8:14]}"
    if len(digits) >= 8:
        return digits[:8]
    cleaned = base.replace("__", "_").strip("_")
    return cleaned or "adapted_run"


def read_entries(path: Path) -> List[Entry]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = {"name", "type", "execution_times_ms"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")

        entries: List[Entry] = []
        for row in reader:
            name = (row.get("name") or "").strip()
            dataset_size = (row.get("type") or "").strip()
            raw_times = (row.get("execution_times_ms") or "").strip()
            if not name or not dataset_size or not raw_times:
                continue
            tokens = [tok for tok in raw_times.replace(",", " ").split(" ") if tok]
            try:
                times_ms = [float(tok) for tok in tokens]
            except ValueError as exc:
                raise ValueError(f"Failed to parse execution times for {name!r}") from exc
            entries.append(Entry(name, dataset_size, times_ms))
    if not entries:
        raise ValueError(f"No usable rows found in {path}")
    return entries


def write_times(entries: Iterable[Entry], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=TIMES_HEADER, lineterminator="\n"
        )
        writer.writeheader()
        for entry in entries:
            for idx, time_s in enumerate(entry.times_s, start=1):
                writer.writerow(
                    {
                        "framework": "tiramisu",
                        "version": "standard",
                        "algorithm": entry.algorithm_normalized,
                        "run": idx,
                        "time_s": f"{time_s:.6f}",
                        "repetition": 1,
                        "dataset_size": entry.dataset_size,
                    }
                )


def write_validation(entries: Iterable[Entry], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=VALIDATION_HEADER, lineterminator="\n"
        )
        writer.writeheader()
        for entry in entries:
            times_s = entry.times_s
            best_time = f"{min(times_s):.6f}" if times_s else ""
            writer.writerow(
                {
                    "framework": "tiramisu",
                    "algorithm": entry.algorithm_normalized,
                    "time_s": best_time,
                    "speedup": "",
                    "valid": "valid",
                    "repetition": 1,
                    "dataset_size": entry.dataset_size,
                }
            )


def main() -> None:
    args = parse_args()
    input_csv: Path = args.input_csv
    run_id = args.run_id or derive_run_id(input_csv)

    entries = read_entries(input_csv)

    dataset_sizes = {entry.dataset_size for entry in entries}
    if len(dataset_sizes) != 1:
        raise ValueError(
            f"Expected a single dataset size in the input, found: {', '.join(sorted(dataset_sizes))}"
        )
    dataset_size = dataset_sizes.pop()

    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    times_path = output_dir / f"times_{dataset_size}.csv"
    validation_path = output_dir / f"validation_{dataset_size}.csv"

    write_times(entries, times_path)
    write_validation(entries, validation_path)

    print(f"Wrote {times_path}")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()
