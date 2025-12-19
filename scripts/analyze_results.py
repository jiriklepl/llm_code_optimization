#!/usr/bin/env python3
"""Compute optimization speedups vs translation baselines (configurable, standard-version per-algorithm baselines), include the translation standard version, and plot geometric means across dataset sizes and algorithms."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


BASELINE_FRAMEWORK = "c"
STATUS_TO_BOOL = {
    "valid": True,
    "true": True,
    "invalid": False,
    "false": False,
    "illegal": False,
    "timeout": False,
}


def normalize_valid_column(series: pd.Series) -> pd.Series:
    """Convert status strings to booleans, defaulting to False."""

    return series.astype(str).str.lower().map(STATUS_TO_BOOL).fillna(False)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""

    return pd.read_csv(path)


def infer_dataset_size(path: Path) -> str | None:
    """Infer dataset size from filenames such as times_SMALL.csv."""

    stem = path.stem
    if "_" not in stem:
        return None

    return stem.split("_")[-1]


def collect_prefixed_files(root: Path, prefix: str) -> list[Path]:
    """Return all CSVs under root starting with the given prefix."""

    if root.is_file():
        if root.name.startswith(f"{prefix}_") and root.suffix == ".csv":
            return [root]
        return []

    files = sorted(p for p in root.rglob(f"{prefix}_*.csv") if p.is_file())
    return files


def normalize_cli_args(values: Sequence[str] | None) -> list[str]:
    """Normalize dataset-size CLI inputs into a unique, ordered list."""

    if not values:
        return []
    normalized: list[str] = []
    for raw in values:
        if raw is None:
            continue
        # Allow comma-separated entries for convenience
        for part in str(raw).split(","):
            cleaned = part.strip()
            if cleaned:
                normalized.append(cleaned)
    return list(dict.fromkeys(normalized))


def filter_dataset_sizes(
    df: pd.DataFrame | None, allowed_sizes: Sequence[str] | None
) -> pd.DataFrame | None:
    """Keep only rows whose dataset_size matches the allowed list."""

    if df is None or not allowed_sizes or "dataset_size" not in df.columns:
        return df
    allowed = {str(size) for size in allowed_sizes}
    return df[df["dataset_size"].astype(str).isin(allowed)].copy()


def filter_frameworks(
    df: pd.DataFrame | None, disallowed_frameworks: Sequence[str] | None
) -> pd.DataFrame | None:
    """Keep only rows whose framework does not match the disallowed list."""

    if df is None or not disallowed_frameworks or "framework" not in df.columns:
        return df
    disallowed = {str(framework) for framework in disallowed_frameworks}
    return df[~df["framework"].astype(str).isin(disallowed)].copy()


def select_best_standard_baseline(
    times_df: pd.DataFrame,
    *,
    baseline_framework: str,
    dataset_sizes: Sequence[str],
) -> pd.DataFrame:
    """Pick per-algorithm baseline from the given framework/version=standard covering all dataset sizes.

    Returns rows for the chosen repetition with columns:
    algorithm, dataset_size, baseline_time_s, baseline_framework, baseline_version, baseline_repetition
    """

    required_sizes = {str(size) for size in dataset_sizes}
    if times_df.empty or not required_sizes:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "dataset_size",
                "baseline_time_s",
                "baseline_framework",
                "baseline_version",
                "baseline_repetition",
            ]
        )

    df = times_df.copy()
    df["dataset_size"] = df["dataset_size"].astype(str)
    df["version_norm"] = df["version"].apply(normalize_version_value)
    df = df[
        (df["framework"] == baseline_framework)
        & (df["version_norm"] == "standard")
        & (df["dataset_size"].isin(required_sizes))
    ].copy()
    empty_template = pd.DataFrame(
        columns=[
            "algorithm",
            "dataset_size",
            "baseline_time_s",
            "baseline_framework",
            "baseline_version",
            "baseline_repetition",
        ]
    )
    if df.empty:
        return empty_template

    candidates: list[dict[str, object]] = []
    for (alg, rep), group in df.groupby(["algorithm", "repetition"]):
        sizes_present = set(group["dataset_size"])
        if not required_sizes.issubset(sizes_present):
            continue
        gmean_time = geometric_mean(group["mean_time_s"])
        if not np.isfinite(gmean_time):
            continue
        candidates.append({"algorithm": alg, "repetition": rep, "geom_time": gmean_time})

    if not candidates:
        return empty_template.copy()

    candidate_df = pd.DataFrame.from_records(candidates)
    best_reps = (
        candidate_df.sort_values(["algorithm", "geom_time", "repetition"])
        .groupby("algorithm", as_index=False)
        .first()
    )[["algorithm", "repetition"]]

    chosen = df.merge(best_reps, on=["algorithm", "repetition"], how="inner")
    chosen = chosen[chosen["dataset_size"].isin(required_sizes)].copy()

    chosen["baseline_time_s"] = chosen["mean_time_s"]
    chosen["baseline_framework"] = baseline_framework
    chosen["baseline_version"] = chosen["version"].apply(format_version_label)
    chosen["baseline_repetition"] = chosen["repetition"]
    return chosen[
        [
            "algorithm",
            "dataset_size",
            "baseline_time_s",
            "baseline_framework",
            "baseline_version",
            "baseline_repetition",
        ]
    ]


def merge_baseline_tables(primary: pd.DataFrame, fallback: pd.DataFrame | None) -> pd.DataFrame:
    """Use primary baseline rows, filling missing algorithms from fallback."""

    if primary is None:
        primary = pd.DataFrame(
            columns=[
                "algorithm",
                "dataset_size",
                "baseline_time_s",
                "baseline_framework",
                "baseline_version",
                "baseline_repetition",
            ]
        )
    if fallback is None or fallback.empty:
        return primary.copy()

    primary_algs = set(primary["algorithm"])
    missing = fallback[~fallback["algorithm"].isin(primary_algs)]
    return pd.concat([primary, missing], ignore_index=True)


def load_and_aggregate_times(paths: Iterable[Path], warmup: int = 0) -> pd.DataFrame:
    """Concatenate times files and compute mean time per key."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = read_csv(path)
        inferred_size = infer_dataset_size(path)

        if "dataset_size" not in df.columns:
            if inferred_size is None:
                raise ValueError(f"Missing dataset_size in {path}")
            df["dataset_size"] = inferred_size
        else:
            df["dataset_size"] = df["dataset_size"].fillna(inferred_size)
            if inferred_size is not None:
                df.loc[df["dataset_size"] == "", "dataset_size"] = inferred_size

        required = {"framework", "algorithm", "repetition", "time_s"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        if "version" not in df.columns:
            df["version"] = ""

        cols_to_keep = ["framework", "version", "algorithm", "repetition", "dataset_size", "time_s"]
        if "run" in df.columns:
            cols_to_keep.append("run")

        trimmed = df[cols_to_keep].copy()

        trimmed["framework"] = trimmed["framework"].astype(str).str.strip()
        trimmed["version"] = trimmed["version"].astype(str).str.strip()
        trimmed["algorithm"] = trimmed["algorithm"].astype(str).str.strip()
        trimmed["dataset_size"] = trimmed["dataset_size"].astype(str).str.strip()
        trimmed["repetition"] = pd.to_numeric(trimmed["repetition"], errors="coerce")
        trimmed["time_s"] = pd.to_numeric(trimmed["time_s"], errors="coerce")

        if "run" in trimmed.columns:
            trimmed["run"] = pd.to_numeric(trimmed["run"], errors="coerce")
            if warmup > 0:
                trimmed = trimmed[trimmed["run"] > warmup]
            trimmed = trimmed.drop(columns=["run"])

        trimmed = trimmed.dropna(subset=["repetition", "time_s", "dataset_size"])
        trimmed = trimmed[trimmed["dataset_size"] != ""]
        trimmed["repetition"] = trimmed["repetition"].astype(int)

        frames.append(trimmed)

    if not frames:
        return pd.DataFrame(
            columns=[
                "framework",
                "version",
                "algorithm",
                "repetition",
                "dataset_size",
                "mean_time_s",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(
            ["framework", "version", "algorithm", "repetition", "dataset_size"],
            as_index=False,
        )["time_s"]
        .mean()
        .rename(columns={"time_s": "mean_time_s"})
    )
    return grouped


def load_and_aggregate_validation(paths: Iterable[Path]) -> pd.DataFrame | None:
    """Concatenate validation files and collapse validity flags per key."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = read_csv(path)
        inferred_size = infer_dataset_size(path)

        if "dataset_size" not in df.columns:
            if inferred_size is None:
                continue
            df["dataset_size"] = inferred_size
        else:
            df["dataset_size"] = df["dataset_size"].fillna(inferred_size)
            if inferred_size is not None:
                df.loc[df["dataset_size"] == "", "dataset_size"] = inferred_size

        required = {"framework", "algorithm", "repetition", "valid"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        if "version" not in df.columns:
            df["version"] = np.nan

        keep_cols = ["framework", "version", "algorithm", "repetition", "dataset_size", "valid"]
        has_time_column = "time_s" in df.columns
        if has_time_column:
            keep_cols.append("time_s")

        trimmed = df[keep_cols].copy()
        trimmed["framework"] = trimmed["framework"].astype(str).str.strip()
        trimmed["version"] = trimmed["version"].apply(
            lambda v: v.strip() if isinstance(v, str) else v
        )
        trimmed["algorithm"] = trimmed["algorithm"].astype(str).str.strip()
        trimmed["dataset_size"] = trimmed["dataset_size"].astype(str).str.strip()
        trimmed["repetition"] = pd.to_numeric(trimmed["repetition"], errors="coerce")

        trimmed["valid"] = normalize_valid_column(trimmed["valid"])

        trimmed["has_time_s"] = False
        if has_time_column:
            trimmed["time_s"] = pd.to_numeric(trimmed["time_s"], errors="coerce")
            trimmed["has_time_s"] = trimmed["time_s"].notna()
            trimmed = trimmed.drop(columns=["time_s"])

        trimmed = trimmed.dropna(subset=["repetition", "dataset_size"])
        trimmed = trimmed[trimmed["dataset_size"] != ""]
        trimmed["repetition"] = trimmed["repetition"].astype(int)

        frames.append(trimmed)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    if "version" in combined.columns:
        if combined["version"].isna().all():
            combined = combined.drop(columns=["version"])
        elif combined["version"].replace("", np.nan).isna().all():
            combined = combined.drop(columns=["version"])

    key_cols = [
        col
        for col in ["framework", "version", "algorithm", "repetition", "dataset_size"]
        if col in combined.columns
    ]
    grouped = (
        combined.groupby(key_cols, as_index=False)
        .agg(valid=("valid", "all"), has_time_s=("has_time_s", "any"))
    )
    return grouped


def apply_validation_filter(
    times: pd.DataFrame, validation: pd.DataFrame | None
) -> pd.DataFrame:
    """Keep only rows that have a matching valid=True entry in validation."""

    if validation is None or validation.empty:
        return times

    join_keys = [
        col
        for col in ["framework", "version", "algorithm", "repetition", "dataset_size"]
        if col in validation.columns
    ]
    if not join_keys:
        return times

    merged = times.merge(validation[join_keys + ["valid"]], on=join_keys, how="left")
    filtered = merged[merged["valid"] == True].drop(columns=["valid"])  # noqa: E712
    # If c is not present in validation, assume valid=True for all its entries
    if not "c" in validation["framework"].values:
        filtered = pd.concat(
            [
                filtered,
                times[times["framework"] == "c"],
            ],
            ignore_index=True,
        )
    return filtered


def geometric_mean(values: Sequence[float]) -> float:
    """Compute the geometric mean of positive values; return NaN when empty."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]

    if len(arr[~np.isfinite(arr) | (arr <= 0)]) > 0:
        raise ValueError("Geometric mean is only defined for positive finite values")

    if arr.size == 0:
        return math.nan
    return float(np.exp(np.log(arr).mean()))


def collect_polybench_algorithm_order(polybench_root: Path) -> dict[str, int]:
    """Return an ordering map for algorithms following the PolybenchC tree."""

    if not polybench_root.exists() or not polybench_root.is_dir():
        return {}

    paths = sorted(
        p
        for p in polybench_root.rglob("*.c")
        if "utilities" not in p.parts  # skip helper sources
    )
    order: dict[str, int] = {}
    for idx, path in enumerate(paths):
        stem = path.stem
        order.setdefault(stem, idx)
    return order


def collect_polybench_categories(polybench_root: Path) -> dict[str, str]:
    """Map each algorithm to its Polybench category for tree-based ordering."""

    if not polybench_root.exists() or not polybench_root.is_dir():
        return {}

    paths = sorted(
        p
        for p in polybench_root.rglob("*.c")
        if "utilities" not in p.parts and "build" not in p.parts
    )
    categories: dict[str, str] = {}
    for path in paths:
        rel_parts = path.relative_to(polybench_root).parts
        if len(rel_parts) < 2:
            continue
        if rel_parts[0] == "linear-algebra" and len(rel_parts) >= 3:
            category = "/".join(rel_parts[:2])
        else:
            category = rel_parts[0]
        categories.setdefault(path.stem, category)
    return categories


def gather_all_algorithms(*dfs: pd.DataFrame) -> list[str]:
    """Collect a stable, deduplicated list of algorithms from multiple DataFrames."""

    algs: list[str] = []
    for df in dfs:
        if df is None or df.empty or "algorithm" not in df.columns:
            continue
        algs.extend(df["algorithm"].astype(str))
    # Preserve first-seen order while removing duplicates
    return list(dict.fromkeys(algs))


def compute_algorithm_order(
    algorithms: Sequence[str],
    order_df: pd.DataFrame,
    ordering: str,
    tree_order: dict[str, int],
    all_algorithms: Sequence[str] | None = None,
) -> list[str]:
    """Return algorithms ordered according to the requested strategy."""

    if all_algorithms is not None:
        alg_list = list(dict.fromkeys(all_algorithms))
    else:
        alg_list = list(dict.fromkeys(algorithms))
    if ordering == "alpha":
        return sorted(alg_list)

    if ordering == "tree":
        def key_fn1(alg: str) -> tuple[int, int | None, str]:
            if alg in tree_order:
                return (0, tree_order[alg], alg)
            return (1, None, alg)

        return sorted(alg_list, key=key_fn1)

    # ordering == "max"
    if order_df is None or order_df.empty or "geom_speedup" not in order_df.columns:
        max_by_alg = pd.Series(dtype=float)
    else:
        max_by_alg = order_df.groupby("algorithm")["geom_speedup"].max()

    def key_fn2(alg: str) -> tuple[float, str]:
        val = max_by_alg.get(alg, math.nan)
        if pd.isna(val):
            return (math.inf, alg)
        return (-float(val), alg)

    return sorted(alg_list, key=key_fn2)


CATEGORY_ORDER = [
    ("invalid_code", "Invalid code"),
    ("invalid_algorithm", "Invalid algorithm"),
    ("slowdown", "Slowdown (<1x)"),
    ("speedup_under", "Speedup < threshold"),
    ("speedup_over", "Speedup >= threshold"),
]


def derive_expected_dataset_sizes(translation_times: pd.DataFrame) -> set[str]:
    """Expected dataset sizes based on translation runs."""

    return set(translation_times["dataset_size"].unique())


def translation_repetitions_by_algorithm(translation_times: pd.DataFrame) -> dict[str, set[int]]:
    """Map each algorithm to repetitions seen in translation runs."""

    reps: dict[str, set[int]] = {}
    for alg, group in translation_times.groupby("algorithm"):
        reps[alg] = set(group["repetition"].unique())
    return reps


def classify_repetition_outcomes(
    translation_times: pd.DataFrame,
    optimization_times: pd.DataFrame,
    optimization_validation: pd.DataFrame | None,
    speedups: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Classify each repetition into mutually exclusive outcome categories."""

    translation_reps = translation_repetitions_by_algorithm(translation_times)

    times_norm = optimization_times.copy()
    times_norm["version"] = times_norm["version"].apply(normalize_version_value)

    if optimization_validation is not None and "version" not in optimization_validation.columns:
        validation_df = optimization_validation.copy()
        validation_df["version"] = ""
    else:
        validation_df = optimization_validation.copy() if optimization_validation is not None else None

    if validation_df is not None:
        validation_df["version"] = validation_df["version"].apply(normalize_version_value)
        validation_norm = validation_df
    else:
        validation_norm = None

    expected_sizes_map: dict[tuple[str, str, str], set[str]] = {}
    for df_src in [times_norm, validation_norm] if validation_norm is not None else [times_norm]:
        for (fw, ver, alg), group in df_src.groupby(["framework", "version", "algorithm"], dropna=False):
            key = (fw, ver, alg)
            expected_sizes_map.setdefault(key, set()).update(group["dataset_size"].astype(str))

    group_keys: set[tuple[str, str, str]] = set()
    if not times_norm.empty:
        group_keys.update(
            (fw, ver, alg)
            for fw, ver, alg in times_norm[["framework", "version", "algorithm"]].itertuples(index=False, name=None)
        )
    if validation_norm is not None and not validation_norm.empty:
        group_keys.update(
            (fw, ver, alg)
            for fw, ver, alg in validation_norm[["framework", "version", "algorithm"]].itertuples(
                index=False, name=None
            )
        )

    records: list[dict[str, object]] = []
    for fw, ver, alg in sorted(group_keys):
        reps_from_times = set(
            times_norm[
                (times_norm["framework"] == fw)
                & (times_norm["version"] == ver)
                & (times_norm["algorithm"] == alg)
            ]["repetition"].unique()
        )
        reps_from_validation: set[int] = set()
        if validation_norm is not None:
            reps_from_validation = set(
                validation_norm[
                    (validation_norm["framework"] == fw)
                    & (validation_norm["version"] == ver)
                    & (validation_norm["algorithm"] == alg)
                ]["repetition"].unique()
            )

        rep_candidates: set[int] = set()
        rep_candidates.update(reps_from_times)
        rep_candidates.update(reps_from_validation)
        rep_candidates.update(translation_reps.get(alg, set()))
        if not rep_candidates:
            continue
        for rep in sorted(rep_candidates):
            val_rows = pd.DataFrame()
            if validation_norm is not None:
                val_rows = validation_norm[
                    (validation_norm["framework"] == fw)
                    & (validation_norm["version"] == ver)
                    & (validation_norm["algorithm"] == alg)
                    & (validation_norm["repetition"] == rep)
                ]
            has_invalid_validation = False
            has_validation_time = False
            if validation_norm is not None and not val_rows.empty:
                has_invalid_validation = not val_rows["valid"].all()
                if "has_time_s" in val_rows.columns:
                    has_validation_time = bool(val_rows["has_time_s"].any())
                elif "time_s" in val_rows.columns:
                    has_validation_time = bool(val_rows["time_s"].notna().any())

            time_rows = times_norm[
                (times_norm["framework"] == fw)
                & (times_norm["version"] == ver)
                & (times_norm["algorithm"] == alg)
                & (times_norm["repetition"] == rep)
            ]
            has_time_reports = not time_rows.empty
            has_any_time = has_time_reports or has_validation_time

            if has_invalid_validation:
                category = "invalid_algorithm" if has_any_time else "invalid_code"
                records.append(
                    {
                        "framework": fw,
                        "version": ver,
                        "algorithm": alg,
                        "repetition": rep,
                        "category": category,
                    }
                )
                continue

            if time_rows.empty:
                category = "invalid_code"
                records.append(
                    {
                        "framework": fw,
                        "version": ver,
                        "algorithm": alg,
                        "repetition": rep,
                        "category": category,
                    }
                )
                continue

            present_sizes = set(time_rows["dataset_size"].astype(str).unique())
            expected_sizes = expected_sizes_map.get((fw, ver, alg), present_sizes)
            missing_sizes = expected_sizes - present_sizes

            if missing_sizes:
                category = "invalid_algorithm"
            else:
                rep_speedups = speedups[
                    (speedups["framework"] == fw)
                    & (speedups["version"] == ver)
                    & (speedups["algorithm"] == alg)
                    & (speedups["repetition"] == rep)
                ]
                gmean = geometric_mean(rep_speedups["speedup"]) if not rep_speedups.empty else math.nan
                if not np.isfinite(gmean):
                    category = "invalid_algorithm"
                elif gmean < 1.0:
                    category = "slowdown"
                elif gmean < threshold:
                    category = "speedup_under"
                else:
                    category = "speedup_over"

            records.append(
                {
                    "framework": fw,
                    "version": ver,
                    "algorithm": alg,
                    "repetition": rep,
                    "category": category,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["framework", "version", "algorithm", "repetition", "category"],
    )


def compute_category_shares(
    outcomes: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Aggregate classification outcomes into per-algorithm shares."""

    rows: list[dict[str, object]] = []
    if outcomes.empty:
        return pd.DataFrame(
            columns=group_cols + ["algorithm", "category", "share_pct", "repetitions"]
        )

    for group_key, subset in outcomes.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        for alg, alg_df in subset.groupby("algorithm"):
            total = len(alg_df)
            for cat, _ in CATEGORY_ORDER:
                count = int((alg_df["category"] == cat).sum())
                share = (count / total * 100.0) if total > 0 else math.nan
                rows.append(
                    {
                        **key_dict,
                        "algorithm": alg,
                        "category": cat,
                        "share_pct": share,
                        "repetitions": total,
                    }
                )

    return pd.DataFrame.from_records(
        rows, columns=group_cols + ["algorithm", "category", "share_pct", "repetitions"]
    )


def plot_category_shares(
    df: pd.DataFrame,
    *,
    out_path: Path,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str] | None,
    ordering_source: pd.DataFrame,
    all_algorithms: Sequence[str] | None,
):
    """Plot stacked bars of outcome shares per algorithm."""

    algorithms = compute_algorithm_order(
        df["algorithm"].unique(), ordering_source, ordering, tree_order, all_algorithms=all_algorithms
    )
    if not algorithms:
        return

    shares = {(alg, cat): 0.0 for alg in algorithms for cat, _ in CATEGORY_ORDER}
    for (alg, cat), subset in df.groupby(["algorithm", "category"]):
        key = (alg, cat)
        if key in shares:
            shares[key] = float(subset["share_pct"].iloc[0])

    x_positions = np.arange(len(algorithms))
    fig, ax = plt.subplots(figsize=(max(8, len(algorithms) * 0.7), 5))

    bottoms = np.zeros(len(algorithms))
    color_map = {
        "invalid_code": "#9e9e9e",
        "invalid_algorithm": "#d62728",
        "slowdown": "#ff7f0e",
        "speedup_under": "#ffd92f",
        "speedup_over": "#2ca02c",
    }

    for cat, label in CATEGORY_ORDER:
        heights = np.array([shares.get((alg, cat), 0.0) for alg in algorithms])
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=color_map.get(cat, "#cccccc"),
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        bottoms += heights

    if ordering == "tree":
        add_category_boundaries(ax, algorithms, tree_categories)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of repetitions (%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_category_boundaries(
    ax: plt.Axes, algorithms: Sequence[str], tree_categories: dict[str, str] | None
) -> None:
    """Draw dashed separators when Polybench categories change."""

    if not tree_categories or not algorithms:
        return

    last = tree_categories.get(algorithms[0])
    for idx, alg in enumerate(algorithms[1:], start=1):
        current = tree_categories.get(alg)
        if current is None or last is None:
            last = current
            continue
        if current != last:
            ax.axvline(idx - 0.5, color="#b0b0b0", linestyle="--", linewidth=0.8, zorder=0)
            last = current


def compute_speedups(
    translation_times: pd.DataFrame,
    optimization_times: pd.DataFrame,
    baseline_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute speedups per key using a provided baseline table or the default C baseline."""

    if baseline_table is not None:
        baseline_df = baseline_table.copy()
    else:
        baseline_runs = translation_times[translation_times["framework"] == BASELINE_FRAMEWORK]
        if baseline_runs.empty:
            raise ValueError(f"No baseline entries for framework '{BASELINE_FRAMEWORK}'")
        baseline_df = baseline_runs.copy()
        baseline_df["baseline_time_s"] = baseline_df["mean_time_s"]
        baseline_df["baseline_framework"] = BASELINE_FRAMEWORK

    baseline_df["dataset_size"] = baseline_df["dataset_size"].astype(str)
    optimization_times = optimization_times.copy()
    optimization_times["dataset_size"] = optimization_times["dataset_size"].astype(str)

    dataset_sizes = sorted(
        set(baseline_df["dataset_size"]).intersection(set(optimization_times["dataset_size"]))
    )

    frames: list[pd.DataFrame] = []
    for size in dataset_sizes:
        base_df = baseline_df[baseline_df["dataset_size"] == size]
        if base_df.empty:
            continue

        baseline_by_algo = (
            base_df.groupby(["algorithm", "dataset_size"], as_index=False).agg(
                baseline_time_s=("baseline_time_s", "mean"),
                baseline_framework=("baseline_framework", "first"),
            )
        )

        opt_df = optimization_times[optimization_times["dataset_size"] == size].rename(
            columns={"mean_time_s": "optimized_time_s"}
        )
        merged = opt_df.merge(
            baseline_by_algo, on=["algorithm", "dataset_size"], how="inner"
        )
        merged["speedup"] = merged["baseline_time_s"] / merged["optimized_time_s"]
        merged = merged[np.isfinite(merged["speedup"]) & (merged["speedup"] > 0)]
        frames.append(merged)

    if not frames:
        return pd.DataFrame(
            columns=[
                "framework",
                "version",
                "algorithm",
                "repetition",
                "dataset_size",
                "speedup",
                "optimized_time_s",
                "baseline_time_s",
                "baseline_framework",
            ]
        )

    result = pd.concat(frames, ignore_index=True)
    column_order = [
        "framework",
        "version",
        "algorithm",
        "repetition",
        "dataset_size",
        "speedup",
        "optimized_time_s",
        "baseline_time_s",
        "baseline_framework",
    ]
    return result[column_order]


def jitter_positions(count: int, spread: float = 0.35) -> np.ndarray:
    """Small symmetric offsets used for plot separation."""

    if count <= 1:
        return np.array([0.0])
    return np.linspace(-spread / 2, spread / 2, count)


def plot_algorithms(
    df: pd.DataFrame,
    *,
    plot_kind: str,
    color_field: str,
    color_label: str,
    out_path: Path,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str] | None,
    all_algorithms: Sequence[str] | None,
    y_min: float | None,
    y_max: float | None,
    log_scale: bool = False,
):
    """Plot per-algorithm values with scatter, bar, or box style."""

    algorithms = compute_algorithm_order(
        df["algorithm"].unique(), df, ordering, tree_order, all_algorithms=all_algorithms
    )
    if not algorithms:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(algorithms) * 0.7), 5))

    palette_values = sorted(df[color_field].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {val: cmap(i % 10) for i, val in enumerate(palette_values)}

    for idx, alg in enumerate(algorithms):
        subset = df[df["algorithm"] == alg].sort_values(color_field)
        if subset.empty:
            continue
        if plot_kind == "box":
            if color_field == "repetition":
                values = subset["geom_speedup"].dropna().to_numpy()
                if values.size == 0:
                    continue
                face_color = color_map.get(palette_values[0], "#76b7b2")
                ax.boxplot(
                    values,
                    positions=[idx],
                    widths=0.6,
                    patch_artist=True,
                    boxprops={"facecolor": face_color, "color": "#4c4c4c"},
                    medianprops={"color": "#2f4f4f", "linewidth": 1.5},
                    whiskerprops={"color": "#4c4c4c"},
                    capprops={"color": "#4c4c4c"},
                    flierprops={
                        "marker": "o",
                        "markersize": 4,
                        "markerfacecolor": face_color,
                        "markeredgecolor": "#4c4c4c",
                    },
                )
            else:
                offsets = jitter_positions(len(palette_values))
                width = 0.6 / max(len(palette_values), 1)
                for off, val in zip(offsets, palette_values):
                    val_subset = subset[subset[color_field] == val]["geom_speedup"].dropna().to_numpy()
                    if val_subset.size == 0:
                        continue
                    pos = idx + off
                    face_color = color_map.get(val, "#76b7b2")
                    ax.boxplot(
                        val_subset,
                        positions=[pos],
                        widths=width,
                        patch_artist=True,
                        boxprops={"facecolor": face_color, "color": "#4c4c4c"},
                        medianprops={"color": "#2f4f4f", "linewidth": 1.5},
                        whiskerprops={"color": "#4c4c4c"},
                        capprops={"color": "#4c4c4c"},
                        flierprops={
                            "marker": "o",
                            "markersize": 4,
                            "markerfacecolor": face_color,
                            "markeredgecolor": "#4c4c4c",
                        },
                    )
            continue
        offsets = jitter_positions(len(subset))
        if offsets.size != len(subset):
            offsets = np.zeros(len(subset))
        x_positions = idx + offsets
        y_values = subset["geom_speedup"].to_numpy()

        if plot_kind == "bar":
            width = 0.6 / max(len(subset), 1)
            for x, (_, row) in zip(x_positions, subset.iterrows()):
                ax.bar(
                    x,
                    row["geom_speedup"],
                    width=width,
                    color=color_map[row[color_field]],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                )
        else:
            if x_positions.size != y_values.size:
                x_positions = np.full(y_values.shape, idx, dtype=float)
            ax.scatter(
                x_positions,
                y_values,
                s=40,
                c=[color_map[val] for val in subset[color_field]],
                edgecolors="black",
                linewidths=0.4,
            )

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
    if ordering == "tree":
        add_category_boundaries(ax, algorithms, tree_categories)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.set_ylabel("Geom. mean speedup vs translation baseline")

    if plot_kind == "box" and color_field != "repetition":
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[val],
                marker="s",
                linestyle="",
                label=str(val),
            )
            for val in palette_values
        ]
        ax.legend(handles=handles, title=color_label, bbox_to_anchor=(1.02, 1), loc="upper left")
    elif plot_kind != "box":
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[val],
                marker="s" if plot_kind == "bar" else "o",
                linestyle="",
                label=str(val),
            )
            for val in palette_values
        ]
        ax.legend(handles=handles, title=color_label, bbox_to_anchor=(1.02, 1), loc="upper left")

    bottom = y_min
    top = y_max
    if log_scale:
        if bottom is not None and bottom <= 0:
            positive = df["geom_speedup"][df["geom_speedup"] > 0]
            if not positive.empty:
                bottom = max(positive.min() * 0.5, np.nextafter(0, 1))
            else:
                bottom = None
    if bottom is not None or top is not None:
        ax.set_ylim(bottom=bottom, top=top)

    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_size_aggregates(
    df: pd.DataFrame,
    *,
    plot_kind: str,
    color_field: str,
    color_label: str,
    out_path: Path,
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
) -> None:
    """Plot values per dataset size, distinguishing series by the given color field."""

    if df.empty or "dataset_size" not in df.columns or color_field not in df.columns:
        return

    dataset_sizes = list(dict.fromkeys(df["dataset_size"].astype(str)))
    color_values = list(dict.fromkeys(df[color_field]))
    cmap = plt.get_cmap("tab10")
    color_map = {val: cmap(i % 10) for i, val in enumerate(color_values)}

    fig, ax = plt.subplots(figsize=(max(8, len(dataset_sizes) * 0.8), 4.8))

    for idx, size in enumerate(dataset_sizes):
        subset = df[df["dataset_size"].astype(str) == str(size)].sort_values(color_field)
        if subset.empty:
            continue
        if plot_kind == "box":
            if color_field == "repetition":
                values = subset["geom_speedup"].dropna().to_numpy()
                if values.size == 0:
                    continue
                face_color = color_map.get(color_values[0], "#76b7b2")
                ax.boxplot(
                    values,
                    positions=[idx],
                    widths=0.6,
                    patch_artist=True,
                    boxprops={"facecolor": face_color, "color": "#4c4c4c"},
                    medianprops={"color": "#2f4f4f", "linewidth": 1.5},
                    whiskerprops={"color": "#4c4c4c"},
                    capprops={"color": "#4c4c4c"},
                    flierprops={
                        "marker": "o",
                        "markersize": 4,
                        "markerfacecolor": face_color,
                        "markeredgecolor": "#4c4c4c",
                    },
                )
            else:
                offsets = jitter_positions(len(color_values))
                width = 0.6 / max(len(color_values), 1)
                for off, val in zip(offsets, color_values):
                    val_subset = subset[subset[color_field] == val]["geom_speedup"].dropna().to_numpy()
                    if val_subset.size == 0:
                        continue
                    pos = idx + off
                    face_color = color_map.get(val, "#76b7b2")
                    ax.boxplot(
                        val_subset,
                        positions=[pos],
                        widths=width,
                        patch_artist=True,
                        boxprops={"facecolor": face_color, "color": "#4c4c4c"},
                        medianprops={"color": "#2f4f4f", "linewidth": 1.5},
                        whiskerprops={"color": "#4c4c4c"},
                        capprops={"color": "#4c4c4c"},
                        flierprops={
                            "marker": "o",
                            "markersize": 4,
                            "markerfacecolor": face_color,
                            "markeredgecolor": "#4c4c4c",
                        },
                    )
            continue
        offsets = jitter_positions(len(subset))
        if offsets.size != len(subset):
            offsets = np.zeros(len(subset))
        x_positions = idx + offsets
        y_values = subset["geom_speedup"].to_numpy()

        if plot_kind == "bar":
            width = 0.6 / max(len(subset), 1)
            for x, (_, row) in zip(x_positions, subset.iterrows()):
                ax.bar(
                    x,
                    row["geom_speedup"],
                    width=width,
                    color=color_map[row[color_field]],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                )
        else:
            if x_positions.size != y_values.size:
                x_positions = np.full(y_values.shape, idx, dtype=float)
            ax.scatter(
                x_positions,
                y_values,
                s=50,
                c=[color_map[val] for val in subset[color_field]],
                edgecolors="black",
                linewidths=0.5,
            )

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xticks(range(len(dataset_sizes)))
    ax.set_xticklabels(dataset_sizes, rotation=45, ha="right")
    ax.set_ylabel("Geom. mean speedup vs translation baseline")

    if plot_kind == "box" and color_field != "repetition":
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[val],
                marker="s",
                linestyle="",
                label=str(val),
            )
            for val in color_values
        ]
        ax.legend(handles=handles, title=color_label, bbox_to_anchor=(1.02, 1), loc="upper left")
    elif plot_kind != "box":
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[val],
                marker="s" if plot_kind == "bar" else "o",
                linestyle="",
                label=str(val),
            )
            for val in color_values
        ]
        ax.legend(handles=handles, title=color_label, bbox_to_anchor=(1.02, 1), loc="upper left")

    bottom = y_min
    top = y_max
    if log_scale:
        if bottom is not None and bottom <= 0:
            positive = df["geom_speedup"][df["geom_speedup"] > 0]
            if not positive.empty:
                bottom = max(positive.min() * 0.5, np.nextafter(0, 1))
            else:
                bottom = None
    if bottom is not None or top is not None:
        ax.set_ylim(bottom=bottom, top=top)
    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def best_per_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Keep the best repetition (max geom_speedup) per requested grouping."""

    if df.empty:
        return df
    idx = df.groupby(group_cols)["geom_speedup"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_heatmap_table(
    geom_df: pd.DataFrame,
    combined_geom_df: pd.DataFrame,
    *,
    best: str,
    ordering: str,
    tree_order: dict[str, int],
    all_algorithms: Sequence[str],
    count_wins: bool,
    include_standard: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Prepare a pivoted table with geom. means plus row/column aggregates.

    When include_standard is False, translation standard versions are removed.
    """

    base_df = combined_geom_df if best in ("combine", "smart-combine") else geom_df
    if base_df is None or base_df.empty:
        return pd.DataFrame(), [], [], [], []

    base = base_df.copy()
    base["version_norm"] = base["version"].apply(normalize_version_value)
    if not include_standard:
        base = base[base["version_norm"] != "standard"].copy()
    if base.empty:
        return pd.DataFrame(), [], [], [], []

    if best in ("pick", "none"):
        base = best_per_group(base, ["framework", "version", "algorithm"])

    base["row_label"] = base.apply(
        lambda r: f"{r['framework']} - {format_version_label(r['version'])}", axis=1
    )
    row_labels = list(dict.fromkeys(base["row_label"]))
    row_label_to_fw = (
        base.drop_duplicates(subset=["row_label"])
        .set_index("row_label")["framework"]
        .to_dict()
    )
    algorithms = compute_algorithm_order(
        base["algorithm"].unique(),
        base,
        ordering,
        tree_order,
        all_algorithms=all_algorithms,
    )
    if not algorithms:
        return pd.DataFrame(), [], [], [], []

    pivot = (
        base.pivot_table(index="row_label", columns="algorithm", values="geom_speedup", aggfunc="max")
        .reindex(index=row_labels, columns=algorithms)
    )
    pivot.index.name = "framework_version"
    pivot.columns.name = "algorithm"

    pivot_with_rows = pivot.assign(geom_mean=pivot.apply(geometric_mean, axis=1))
    col_geom = pivot_with_rows.apply(geometric_mean, axis=0)
    col_geom.name = "geom_mean"
    full = pd.concat([pivot_with_rows, col_geom.to_frame().T], axis=0)

    row_labels_with_total = row_labels + ["geom_mean"]
    col_labels_with_total = algorithms + ["geom_mean"]
    full = full.reindex(index=row_labels_with_total, columns=col_labels_with_total)

    if count_wins:
        wins_counter = {lbl: 0 for lbl in row_labels}
        for alg in algorithms:
            col_series = full.loc[row_labels, alg].dropna()
            if col_series.empty:
                continue
            winner = col_series.idxmax()
            wins_counter[winner] = wins_counter.get(winner, 0) + 1

        full["wins"] = np.nan
        for lbl in row_labels:
            full.at[lbl, "wins"] = wins_counter.get(lbl, 0)
        col_labels_with_total = col_labels_with_total + ["wins"]
        full = full.reindex(columns=col_labels_with_total)

    row_frameworks = [row_label_to_fw.get(lbl) for lbl in row_labels] + [None]
    return full, row_labels_with_total, col_labels_with_total, algorithms, row_frameworks


def build_correctness_heatmap_table(
    outcomes: pd.DataFrame,
    *,
    ordering: str,
    tree_order: dict[str, int],
    all_algorithms: Sequence[str],
    include_standard: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Prepare a pivoted table of validity shares plus row/column aggregates.

    When include_standard is False, translation standard versions are removed.
    """

    if outcomes is None or outcomes.empty:
        return pd.DataFrame(), [], [], [], []

    base = outcomes.copy()
    base["version_norm"] = base["version"].apply(normalize_version_value)
    if not include_standard:
        base = base[base["version_norm"] != "standard"].copy()
    if base.empty:
        return pd.DataFrame(), [], [], [], []

    base["is_valid"] = ~base["category"].isin(["invalid_code", "invalid_algorithm"])
    grouped = (
        base.groupby(["framework", "version", "algorithm"], dropna=False)
        .agg(valid_reps=("is_valid", "sum"), total_reps=("is_valid", "size"))
        .reset_index()
    )
    grouped["valid_pct"] = grouped.apply(
        lambda r: (r["valid_reps"] / r["total_reps"] * 100.0) if r["total_reps"] > 0 else math.nan,
        axis=1,
    )

    grouped["row_label"] = grouped.apply(
        lambda r: f"{r['framework']} - {format_version_label(r['version'])}", axis=1
    )
    row_labels = list(dict.fromkeys(grouped["row_label"]))
    row_label_to_fw = (
        grouped.drop_duplicates(subset=["row_label"])
        .set_index("row_label")["framework"]
        .to_dict()
    )
    algorithms = compute_algorithm_order(
        grouped["algorithm"].unique(),
        grouped,
        ordering,
        tree_order,
        all_algorithms=all_algorithms,
    )
    if not algorithms:
        return pd.DataFrame(), [], [], [], []

    pivot = (
        grouped.pivot_table(index="row_label", columns="algorithm", values="valid_pct", aggfunc="max")
        .reindex(index=row_labels, columns=algorithms)
    )
    pivot.index.name = "framework_version"
    pivot.columns.name = "algorithm"

    pivot_with_rows = pivot.assign(mean_valid=pivot.mean(axis=1))
    col_mean = pivot_with_rows.mean(axis=0)
    col_mean.name = "mean_valid"
    full = pd.concat([pivot_with_rows, col_mean.to_frame().T], axis=0)

    row_labels_with_total = row_labels + ["mean_valid"]
    col_labels_with_total = algorithms + ["mean_valid"]
    full = full.reindex(index=row_labels_with_total, columns=col_labels_with_total)

    row_frameworks = [row_label_to_fw.get(lbl) for lbl in row_labels] + [None]
    return full, row_labels_with_total, col_labels_with_total, algorithms, row_frameworks


def plot_heatmap_table(
    table: pd.DataFrame,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    out_path: Path,
    ordering: str,
    tree_categories: dict[str, str],
    algorithms: Sequence[str],
    row_frameworks: Sequence[str | None],
    cbar_label: str = "Geom. mean speedup vs translation baseline",
    value_fmt: str = "{:.2f}",
    value_fmt_overrides: dict[str, str] | None = None,
    cmap_name: str = "YlGnBu",
    special_rows: Sequence[str] | None = None,
    special_cols: Sequence[str] | None = None,
    hide_colorbar: bool = False,
) -> None:
    """Plot a color-coded heatmap and highlight per-column winners."""

    if table.empty or not row_labels or not col_labels:
        return

    data = table.to_numpy(dtype=float)
    if not np.isfinite(data).any():
        return
    norm_source = table.drop(columns=["wins"], errors="ignore").to_numpy(dtype=float)
    finite_vals = norm_source[np.isfinite(norm_source)]
    vmin = float(finite_vals.min()) if finite_vals.size else None
    vmax = float(finite_vals.max()) if finite_vals.size else None

    masked = np.ma.masked_invalid(data)
    fig_width = max(8.0, len(col_labels) * 0.8)
    fig_height = max(4.0, len(row_labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    fmt_overrides = {"wins": "{:.0f}", "mean_valid": "{:.0f}"}
    if value_fmt_overrides:
        fmt_overrides.update(value_fmt_overrides)
    special_row_set = set(special_rows or [])
    special_col_set = set(special_cols or [])
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            val = table.iloc[i, j]
            if not np.isfinite(val):
                continue
            norm_val = im.norm(val)
            text_color = "black" if norm_val < 0.6 else "white"
            fmt = fmt_overrides.get(col_label, value_fmt)
            ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    if not hide_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    for idx, label in enumerate(col_labels):
        if label in special_col_set:
            ax.axvline(idx - 0.5, color="#707070", linestyle="-", linewidth=1.1, zorder=1)
    for idx, label in enumerate(row_labels):
        if label in special_row_set:
            ax.axhline(idx - 0.5, color="#707070", linestyle="-", linewidth=1.1, zorder=1)

    main_rows = row_labels[:-1] if len(row_labels) > 1 else row_labels
    for col_idx, col_label in enumerate(col_labels):
        if col_label in special_col_set:
            continue
        col_series = table.loc[main_rows, col_label].dropna()
        if col_series.empty:
            continue
        max_val = col_series.max()
        if not np.isfinite(max_val):
            continue
        winners = col_series.index[
            np.isclose(col_series, max_val, rtol=1e-9, atol=1e-9)
        ]
        for best_row_label in winners:
            if best_row_label not in row_labels:
                continue
            row_idx = row_labels.index(best_row_label)
            ax.add_patch(
                Rectangle(
                    (col_idx - 0.5, row_idx - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=1.5,
                )
            )

    if ordering == "tree" and algorithms:
        add_category_boundaries(ax, algorithms, tree_categories)

    if row_frameworks:
        for idx in range(1, min(len(row_labels), len(row_frameworks))):
            prev_fw = row_frameworks[idx - 1]
            curr_fw = row_frameworks[idx]
            if prev_fw is None or curr_fw is None or prev_fw == curr_fw:
                continue
            ax.axhline(idx - 0.5, color="#b0b0b0", linestyle="--", linewidth=0.8, zorder=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def best_speedups_per_dataset_size(
    speedups: pd.DataFrame, *, include_baseline: bool, label: str
) -> pd.DataFrame:
    """Best speedup per dataset size for each algorithm, optionally allowing the baseline."""

    if speedups.empty:
        return pd.DataFrame(
            columns=["framework", "version", "algorithm", "dataset_size", "repetition", "geom_speedup"]
        )

    candidates = speedups[
        ["framework", "version", "algorithm", "dataset_size", "speedup"]
    ].rename(columns={"speedup": "best_speedup"})
    candidates["dataset_size"] = candidates["dataset_size"].astype(str)
    frames = [candidates]
    if include_baseline:
        baseline = (
            candidates[["framework", "version", "algorithm", "dataset_size"]]
            .drop_duplicates()
            .copy()
        )
        baseline["best_speedup"] = 1.0
        frames.append(baseline)

    best_per_size = (
        pd.concat(frames, ignore_index=True)
        .groupby(["framework", "version", "algorithm", "dataset_size"], as_index=False)[
            "best_speedup"
        ]
        .max()
    )
    best_per_size = best_per_size.rename(columns={"best_speedup": "geom_speedup"})
    best_per_size["repetition"] = label
    return best_per_size[
        ["framework", "version", "algorithm", "dataset_size", "repetition", "geom_speedup"]
    ]


def compute_geom_by_dataset_size(
    speedups: pd.DataFrame, *, best: str, include_baseline: bool
) -> pd.DataFrame:
    """Return geom. means per (algorithm, repetition, dataset_size), respecting best mode."""

    if speedups.empty:
        return pd.DataFrame(
            columns=["framework", "version", "algorithm", "dataset_size", "repetition", "geom_speedup"]
        )

    base = (
        speedups.groupby(
            ["framework", "version", "algorithm", "repetition", "dataset_size"], as_index=False
        )
        .agg(geom_speedup=("speedup", geometric_mean))
        .dropna(subset=["geom_speedup"])
    )
    base["dataset_size"] = base["dataset_size"].astype(str)

    if best == "none":
        return base
    if best == "pick":
        return best_per_group(base, ["framework", "version", "algorithm", "dataset_size"])
    if best in ("combine", "smart-combine"):
        label = "smart-combine" if include_baseline else "combined"
        return best_speedups_per_dataset_size(speedups, include_baseline=include_baseline, label=label)
    return base


def aggregate_over_algorithms_by_size(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate across algorithms (and repetitions) per dataset size with a geom. mean."""

    if df.empty:
        return pd.DataFrame(
            columns=group_cols + ["dataset_size", "geom_speedup", "algorithm_count", "entries"]
        )

    cleaned = df.dropna(subset=["geom_speedup"])
    if cleaned.empty:
        return pd.DataFrame(
            columns=group_cols + ["dataset_size", "geom_speedup", "algorithm_count", "entries"]
        )

    aggregated = (
        cleaned.groupby(group_cols + ["dataset_size"], dropna=False)
        .agg(
            geom_speedup=("geom_speedup", geometric_mean),
            algorithm_count=("algorithm", "nunique"),
            entries=("geom_speedup", "size"),
        )
        .reset_index()
    )
    aggregated["dataset_size"] = aggregated["dataset_size"].astype(str)
    return aggregated


def geom_mean_of_best_speedups(
    speedups: pd.DataFrame, *, include_baseline: bool = False, label: str = "combined"
) -> pd.DataFrame:
    """Geom. mean after taking the best speedup per dataset size.

    When include_baseline is True, a baseline speedup of 1.0 is added as a candidate
    for every present (framework, version, algorithm, dataset_size) tuple.
    """

    if speedups.empty:
        return pd.DataFrame(
            columns=[
                "framework",
                "version",
                "algorithm",
                "repetition",
                "geom_speedup",
                "dataset_sizes",
            ]
        )

    candidates = speedups[
        ["framework", "version", "algorithm", "dataset_size", "speedup"]
    ].rename(columns={"speedup": "best_speedup"})
    frames = [candidates]
    if include_baseline:
        baseline = (
            candidates[["framework", "version", "algorithm", "dataset_size"]]
            .drop_duplicates()
            .copy()
        )
        baseline["best_speedup"] = 1.0
        frames.append(baseline)

    best_per_size = (
        pd.concat(frames, ignore_index=True)
        .groupby(["framework", "version", "algorithm", "dataset_size"], as_index=False)[
            "best_speedup"
        ]
        .max()
    )
    combined = best_per_size.groupby(
        ["framework", "version", "algorithm"], as_index=False
    ).agg(
        geom_speedup=("best_speedup", geometric_mean),
        dataset_sizes=("dataset_size", "nunique"),
    )
    combined["repetition"] = label
    return combined[
        ["framework", "version", "algorithm", "repetition", "geom_speedup", "dataset_sizes"]
    ]


def best_suffix(best: str) -> str:
    """Filename suffix describing the chosen best-selection mode."""

    if best == "none":
        return ""
    if best == "pick":
        return "_best"
    if best == "combine":
        return "_best-combine"
    if best == "smart-combine":
        return "_best-smart-combine"
    return f"_best-{best}"


def append_label_suffix(base: str, label_suffix: str | None) -> str:
    """Append an optional label suffix to a base filename stem."""

    if not label_suffix:
        return base
    cleaned = str(label_suffix).strip("_")
    return f"{base}_{cleaned}" if cleaned else base


def build_label_suffix(*, baseline: str, dataset_sizes: Sequence[str]) -> str:
    """Construct a suffix describing explicit baseline/dataset selections."""

    parts: list[str] = []
    if baseline and baseline != BASELINE_FRAMEWORK:
        parts.append(f"baseline-{baseline}")
    if dataset_sizes:
        joined_sizes = "-".join(str(size) for size in dataset_sizes)
        parts.append(f"datasets-{joined_sizes}")
    return "__".join(parts)


def plot_per_framework_version(
    geom_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    combined_geom_df: pd.DataFrame,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
):
    """Plot geom. means grouped by (framework, version)."""

    for (framework, version), subset in geom_df.groupby(
        ["framework", "version"], dropna=False
    ):
        if best in ("combine", "smart-combine"):
            data = combined_geom_df[
                (combined_geom_df["framework"] == framework)
                & (combined_geom_df["version"] == version)
            ].copy()
        else:
            data = subset.copy()
            if best == "pick":
                data = best_per_group(data, ["algorithm"])
        if data.empty:
            continue

        version_label = format_version_label(version)
        stem = f"fw-{framework}_ver-{version_label}_{plot_kind}{best_suffix(best)}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_algorithms(
            data,
            plot_kind=plot_kind,
            color_field="repetition",
            color_label="Repetition",
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_per_version_across_frameworks(
    geom_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    combined_geom_df: pd.DataFrame,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
):
    """Plot geom. means grouped by version across frameworks."""

    for version, subset in geom_df.groupby("version", dropna=False):
        if best in ("combine", "smart-combine"):
            data = combined_geom_df[combined_geom_df["version"] == version].copy()
        else:
            data = subset.copy()
            if best == "pick":
                data = best_per_group(data, ["algorithm", "framework"])
        if data.empty:
            continue

        version_label = format_version_label(version)
        stem = f"version-{version_label}_{plot_kind}{best_suffix(best)}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_algorithms(
            data,
            plot_kind=plot_kind,
            color_field="framework",
            color_label="Framework",
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_per_framework_across_versions(
    geom_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    combined_geom_df: pd.DataFrame,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
):
    """Plot geom. means grouped by framework, coloring by version."""

    for framework, subset in geom_df.groupby("framework", dropna=False):
        if best in ("combine", "smart-combine"):
            data = combined_geom_df[combined_geom_df["framework"] == framework].copy()
        else:
            data = subset.copy()
            data["version_label"] = data["version"].apply(format_version_label)
            if best == "pick":
                data = best_per_group(data, ["algorithm", "version_label"])
        if data.empty:
            continue

        if best in ("combine", "smart-combine"):
            data["version_label"] = data["version"].apply(format_version_label)
        stem = f"framework-{framework}_versions_{plot_kind}{best_suffix(best)}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_algorithms(
            data,
            plot_kind=plot_kind,
            color_field="version_label",
            color_label="Version",
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_per_framework_version_agg_algorithms(
    per_size_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    out_dir: Path,
    label_suffix: str,
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
) -> None:
    """Plot geom. means aggregated over algorithms per dataset size for each (framework, version)."""

    for (framework, version), subset in per_size_df.groupby(
        ["framework", "version"], dropna=False
    ):
        if subset.empty:
            continue

        aggregated = aggregate_over_algorithms_by_size(subset, ["framework", "version"])
        if aggregated.empty:
            continue

        aggregated["version_label"] = aggregated["version"].apply(format_version_label)
        version_label = aggregated["version_label"].iloc[0]
        stem = f"fw-{framework}_ver-{version_label}_{plot_kind}{best_suffix(best)}_agg-alg"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_dataset_size_aggregates(
            aggregated,
            plot_kind=plot_kind,
            color_field="version_label",
            color_label="Version",
            out_path=out_dir / file_name,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_per_version_across_frameworks_agg_algorithms(
    per_size_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    out_dir: Path,
    label_suffix: str,
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
) -> None:
    """Plot geom. means aggregated over algorithms grouped by version across frameworks."""

    for version, subset in per_size_df.groupby("version", dropna=False):
        if subset.empty:
            continue

        aggregated = aggregate_over_algorithms_by_size(subset, ["version"])
        if aggregated.empty:
            continue

        aggregated["version_label"] = aggregated["version"].apply(format_version_label)
        version_label = format_version_label(version)
        stem = f"version-{version_label}_frameworks_{plot_kind}{best_suffix(best)}_agg-alg"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_dataset_size_aggregates(
            aggregated,
            plot_kind=plot_kind,
            color_field="version_label",
            color_label="Version",
            out_path=out_dir / file_name,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_per_framework_across_versions_agg_algorithms(
    per_size_df: pd.DataFrame,
    *,
    plot_kind: str,
    best: str,
    out_dir: Path,
    label_suffix: str,
    y_min: float | None,
    y_max: float | None,
    log_scale: bool,
    plot_format: str,
) -> None:
    """Plot geom. means aggregated over algorithms grouped by framework across versions."""

    for framework, subset in per_size_df.groupby("framework", dropna=False):
        if subset.empty:
            continue

        data = subset.copy()
        data["version_label"] = data["version"].apply(format_version_label)
        aggregated = aggregate_over_algorithms_by_size(data, ["framework", "version", "version_label"])
        if aggregated.empty:
            continue

        stem = f"framework-{framework}_versions_{plot_kind}{best_suffix(best)}_agg-alg"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        plot_dataset_size_aggregates(
            aggregated,
            plot_kind=plot_kind,
            color_field="version_label",
            color_label="Version",
            out_path=out_dir / file_name,
            y_min=y_min,
            y_max=y_max,
            log_scale=log_scale,
        )


def plot_shares_per_framework_version(
    shares_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    *,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    threshold: float,
    plot_format: str,
):
    """Plot outcome shares grouped by (framework, version)."""

    for (framework, version), subset in shares_df.groupby(["framework", "version"], dropna=False):
        if subset.empty:
            continue
        version_label = format_version_label(version)
        stem = f"fw-{framework}_ver-{version_label}_shares_thr-{threshold:.2f}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        order_source = geom_df[
            (geom_df["framework"] == framework) & (geom_df["version"] == version)
        ]
        plot_category_shares(
            subset,
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            ordering_source=order_source if not order_source.empty else subset,
        )


def plot_shares_per_version(
    shares_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    *,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    threshold: float,
    plot_format: str,
):
    """Plot outcome shares grouped by version across frameworks."""

    for version, subset in shares_df.groupby("version", dropna=False):
        if subset.empty:
            continue
        version_label = format_version_label(version)
        stem = f"version-{version_label}_shares_thr-{threshold:.2f}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        order_source = geom_df[geom_df["version"] == version]
        plot_category_shares(
            subset,
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            ordering_source=order_source if not order_source.empty else subset,
        )


def plot_shares_per_framework(
    shares_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    *,
    out_dir: Path,
    label_suffix: str,
    ordering: str,
    tree_order: dict[str, int],
    tree_categories: dict[str, str],
    all_algorithms: Sequence[str],
    threshold: float,
    plot_format: str,
):
    """Plot outcome shares grouped by framework across versions."""

    for framework, subset in shares_df.groupby("framework", dropna=False):
        if subset.empty:
            continue
        stem = f"framework-{framework}_shares_thr-{threshold:.2f}"
        file_name = f"{append_label_suffix(stem, label_suffix)}.{plot_format}"
        order_source = geom_df[geom_df["framework"] == framework]
        plot_category_shares(
            subset,
            out_path=out_dir / file_name,
            ordering=ordering,
            tree_order=tree_order,
            tree_categories=tree_categories,
            all_algorithms=all_algorithms,
            ordering_source=order_source if not order_source.empty else subset,
        )
def format_version_label(version: object) -> str:
    """Return a stable string label for version values."""

    normalized = normalize_version_value(version)
    return normalized if normalized else "noversion"


def normalize_version_value(version: object) -> str:
    """Normalize version values for grouping/joins."""

    if isinstance(version, str):
        cleaned = version.strip()
        if cleaned == "from_model":
            return "from_plan"
        return cleaned
    if pd.isna(version):
        return ""
    return str(version).strip()


def normalize_version_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a copy with 'version' normalized (if present)."""

    if df is None or df.empty or "version" not in df.columns:
        return df
    out = df.copy()
    out["version"] = out["version"].apply(normalize_version_value)
    return out


def extract_standard_runs(translation_times: pd.DataFrame) -> pd.DataFrame:
    """Return translation rows marked as the standard version."""

    if translation_times.empty or "version" not in translation_times.columns:
        return translation_times.iloc[0:0].copy()

    mask = translation_times["version"].apply(normalize_version_value) == "standard"
    return translation_times[mask].copy()


def append_standard_version(
    optimization_times: pd.DataFrame, standard_runs: pd.DataFrame
) -> pd.DataFrame:
    """Append translation standard runs (deduped) to the optimization table."""

    if standard_runs.empty:
        return optimization_times

    combined = pd.concat([optimization_times, standard_runs], ignore_index=True)
    return combined.drop_duplicates(
        subset=["framework", "version", "algorithm", "repetition", "dataset_size"],
        keep="first",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute speedups of optimization runs versus translation baselines and plot "
            "geometric means aggregated across dataset sizes (including translation standard) "
            "along with companion plots aggregated over algorithms."
        )
    )
    parser.add_argument(
        "translation_path",
        help="Directory (or file tree) containing translation times_*.csv and validation_*.csv files.",
    )
    parser.add_argument(
        "optimization_path",
        help="Directory (or file tree) containing optimization times_*.csv and validation_*.csv files.",
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_FRAMEWORK,
        help=(
            "Framework to use as the baseline (requires version='standard' data). "
            "Per algorithm, the best standard repetition covering all dataset sizes is chosen; "
            "if unavailable, falls back to 'c' for that algorithm."
        ),
    )
    parser.add_argument(
        "--plot",
        choices=["bar", "scatter", "box", "stacked"],
        default="bar",
        help=(
            "Plot style to use: bar/scatter for speedups, box to collapse repetitions "
            "into a single box per algorithm, stacked for outcome shares."
        ),
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=1.1,
        help="Threshold separating modest vs strong speedups in outcome-share plots.",
    )
    parser.add_argument(
        "--ordering",
        choices=["max", "tree", "alpha"],
        default="alpha",
        help=(
            "How to order algorithms on the x-axis: max = descending by highest observed "
            "geom. speedup, tree = PolybenchC folder order, alpha = alphabetical."
        ),
    )
    parser.add_argument(
        "--dataset-size",
        dest="dataset_sizes",
        action="append",
        help=(
            "Restrict processing to specific dataset sizes (e.g., SMALL). "
            "Can be repeated or comma-separated to select multiple sizes."
        ),
    )
    parser.add_argument(
        "--no-framework",
        action="append",
        help=(
            "Do not process the specified frameworks (e.g., tiramisu)"
        )
    )
    parser.add_argument(
        "--best",
        choices=["none", "pick", "combine", "smart-combine"],
        default="none",
        help=(
            "How to handle multiple repetitions: none = plot all repetitions; pick = keep "
            "the best repetition per grouping (current behavior); combine = geom. mean of the "
            "best speedups per dataset size; smart-combine = like combine but also allow "
            "the translation baseline (1.0x) to win per dataset size."
        ),
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots/translation_vs_optimization",
        help="Where to store the speedup CSV and generated plots.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for the plots.",
    )
    parser.add_argument(
        "--count-wins",
        action="store_true",
        help="Add win tallies per framework/version to the heatmap CSV/plot.",
    )
    parser.add_argument(
        "--standard-heat",
        action="store_true",
        help="Include translation standard version rows in the speedup and correctness heatmaps.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower limit for plot y-axis.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper limit for plot y-axis.",
    )
    parser.add_argument(
        "--polybench-root",
        default="PolybenchC-4.2.1",
        help="Path to the PolybenchC tree (used for tree ordering).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output format for the generated plots.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of initial runs to discard as warmup (default: 0).",
    )
    parser.add_argument(
        "--hide-colorbar",
        action="store_true",
        help="Do not include colorbars in heatmap plots.",
    )

    args = parser.parse_args()
    if args.best_only:
        args.best = "pick"

    dataset_size_filter = normalize_cli_args(args.dataset_sizes)
    framework_filter = normalize_cli_args(args.no_framework)

    translation_root = Path(args.translation_path)
    optimization_root = Path(args.optimization_path)
    polybench_root = Path(args.polybench_root)
    plot_format = args.format

    if not translation_root.exists():
        raise FileNotFoundError(f"Translation path {translation_root} does not exist.")
    if not optimization_root.exists():
        raise FileNotFoundError(f"Optimization path {optimization_root} does not exist.")

    translation_time_files = collect_prefixed_files(translation_root, "times")
    optimization_time_files = collect_prefixed_files(optimization_root, "times")
    if not translation_time_files:
        raise FileNotFoundError(f"No times_*.csv files found under {translation_root}")
    if not optimization_time_files:
        raise FileNotFoundError(f"No times_*.csv files found under {optimization_root}")

    translation_val_files = collect_prefixed_files(translation_root, "validation")
    optimization_val_files = collect_prefixed_files(optimization_root, "validation")

    translation_times = load_and_aggregate_times(translation_time_files, warmup=args.warmup)
    optimization_times = load_and_aggregate_times(optimization_time_files, warmup=args.warmup)

    translation_validation = (
        load_and_aggregate_validation(translation_val_files) if translation_val_files else None
    )
    optimization_validation = (
        load_and_aggregate_validation(optimization_val_files) if optimization_val_files else None
    )

    translation_times = normalize_version_column(translation_times)
    optimization_times = normalize_version_column(optimization_times)
    translation_validation = normalize_version_column(translation_validation)
    optimization_validation = normalize_version_column(optimization_validation)

    if framework_filter:
        print(f"Filtering out frameworks: {', '.join(framework_filter)}")
        translation_times = filter_frameworks(translation_times, framework_filter)
        optimization_times = filter_frameworks(optimization_times, framework_filter)
        translation_validation = filter_frameworks(translation_validation, framework_filter)
        optimization_validation = filter_frameworks(optimization_validation, framework_filter)

    # Keep full copies for classification to ensure validity checks consider all dataset sizes
    translation_times_full = translation_times.copy()
    optimization_times_full = optimization_times.copy()
    optimization_validation_full = optimization_validation.copy() if optimization_validation is not None else None

    if dataset_size_filter:
        print(f"Filtering to dataset sizes: {', '.join(dataset_size_filter)}")
        translation_times = filter_dataset_sizes(translation_times, dataset_size_filter)
        optimization_times = filter_dataset_sizes(optimization_times, dataset_size_filter)
        translation_validation = filter_dataset_sizes(translation_validation, dataset_size_filter)
        optimization_validation = filter_dataset_sizes(optimization_validation, dataset_size_filter)

    optimization_times_raw = optimization_times.copy()

    translation_times = apply_validation_filter(translation_times, translation_validation)
    optimization_times = apply_validation_filter(optimization_times, optimization_validation)

    standard_runs = extract_standard_runs(translation_times)
    optimization_times = append_standard_version(optimization_times, standard_runs)
    optimization_times_raw = append_standard_version(optimization_times_raw, standard_runs)

    baseline_frameworks = {args.baseline}
    if args.baseline != BASELINE_FRAMEWORK:
        baseline_frameworks.add(BASELINE_FRAMEWORK)

    baseline_sources = []
    if not translation_times.empty:
        baseline_sources.append(translation_times)
    if not optimization_times.empty:
        baseline_sources.append(optimization_times)
    baseline_source_df = (
        pd.concat(baseline_sources, ignore_index=True) if baseline_sources else translation_times
    )

    baseline_candidates = baseline_source_df[
        baseline_source_df["framework"].isin(baseline_frameworks)
        & (baseline_source_df["version"].apply(normalize_version_value) == "standard")
    ]
    baseline_sizes = set(baseline_candidates["dataset_size"].astype(str))
    optimization_sizes = set(optimization_times["dataset_size"].astype(str))
    considered_sizes = sorted(baseline_sizes.intersection(optimization_sizes))
    if not considered_sizes:
        raise RuntimeError("No overlapping dataset sizes between translation and optimization runs.")

    primary_baseline = select_best_standard_baseline(
        baseline_source_df, baseline_framework=args.baseline, dataset_sizes=considered_sizes
    )
    fallback_baseline = (
        select_best_standard_baseline(
            baseline_source_df, baseline_framework=BASELINE_FRAMEWORK, dataset_sizes=considered_sizes
        )
        if args.baseline != BASELINE_FRAMEWORK
        else pd.DataFrame(
            columns=[
                "algorithm",
                "dataset_size",
                "baseline_time_s",
                "baseline_framework",
                "baseline_version",
                "baseline_repetition",
            ]
        )
    )
    baseline_table = merge_baseline_tables(primary_baseline, fallback_baseline)
    if baseline_table.empty:
        raise RuntimeError("No valid baseline repetitions found for the requested framework(s).")

    speedups = compute_speedups(translation_times, optimization_times, baseline_table=baseline_table)
    if speedups.empty:
        raise RuntimeError("No overlapping data found to compute speedups.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_suffix = build_label_suffix(
        baseline=args.baseline,
        dataset_sizes=dataset_size_filter,
    )
    csv_stem = append_label_suffix("speedups", label_suffix)
    csv_path = out_dir / f"{csv_stem}.csv"
    speedups.sort_values(
        ["dataset_size", "algorithm", "framework", "version", "repetition"]
    ).to_csv(csv_path, index=False)
    print(f"Wrote speedup table to {csv_path}")

    outcomes = classify_repetition_outcomes(
        translation_times_full,
        optimization_times_full,
        optimization_validation_full,
        speedups,
        threshold=args.speedup_threshold,
    )

    # Filter outcomes to only include repetitions present in the requested dataset sizes
    if dataset_size_filter:
        relevant_groups = optimization_times[["framework", "version", "algorithm"]].drop_duplicates()
        if optimization_validation is not None:
            val_groups = optimization_validation[["framework", "version", "algorithm"]].drop_duplicates()
            relevant_groups = pd.concat([relevant_groups, val_groups], ignore_index=True).drop_duplicates()

        outcomes = outcomes.merge(relevant_groups, on=["framework", "version", "algorithm"], how="inner")

    outcomes_stem = append_label_suffix(
        f"outcomes_thr-{args.speedup_threshold:.2f}", label_suffix
    )
    outcomes_csv = out_dir / f"{outcomes_stem}.csv"
    outcomes.sort_values(["framework", "version", "algorithm", "repetition"]).to_csv(outcomes_csv, index=False)
    print(f"Wrote outcome categories to {outcomes_csv}")

    outcomes_for_shares = outcomes[outcomes["version"].apply(normalize_version_value) != "standard"].copy()
    shares_fw_ver = compute_category_shares(outcomes_for_shares, ["framework", "version"])
    shares_ver = compute_category_shares(outcomes_for_shares, ["version"])
    shares_fw = compute_category_shares(outcomes_for_shares, ["framework"])
    shares_fw_ver_stem = append_label_suffix(
        f"shares_fw_ver_thr-{args.speedup_threshold:.2f}", label_suffix
    )
    shares_ver_stem = append_label_suffix(
        f"shares_version_thr-{args.speedup_threshold:.2f}", label_suffix
    )
    shares_fw_stem = append_label_suffix(
        f"shares_framework_thr-{args.speedup_threshold:.2f}", label_suffix
    )
    shares_fw_ver_csv = out_dir / f"{shares_fw_ver_stem}.csv"
    shares_ver_csv = out_dir / f"{shares_ver_stem}.csv"
    shares_fw_csv = out_dir / f"{shares_fw_stem}.csv"
    shares_fw_ver.sort_values(["framework", "version", "algorithm", "category"]).to_csv(
        shares_fw_ver_csv, index=False
    )
    shares_ver.sort_values(["version", "algorithm", "category"]).to_csv(shares_ver_csv, index=False)
    shares_fw.sort_values(["framework", "algorithm", "category"]).to_csv(shares_fw_csv, index=False)
    print(f"Wrote share tables to {shares_fw_ver_csv}, {shares_ver_csv}, {shares_fw_csv}")

    speedup_plot_kind = "bar" if args.plot == "stacked" else args.plot
    valid_keys = outcomes[
        ~outcomes["category"].isin(["invalid_code", "invalid_algorithm"])
    ][["framework", "version", "algorithm", "repetition"]].drop_duplicates()
    speedups_for_plots = speedups.copy()
    speedups_for_plots["version"] = speedups_for_plots["version"].apply(normalize_version_value)
    if not valid_keys.empty:
        speedups_for_plots = speedups_for_plots.merge(
            valid_keys, on=["framework", "version", "algorithm", "repetition"], how="inner"
        )
    else:
        speedups_for_plots = speedups_for_plots.iloc[0:0]

    geom_df = (
        speedups_for_plots.groupby(
            ["framework", "version", "algorithm", "repetition"], as_index=False
        )
        .agg(
            geom_speedup=("speedup", geometric_mean),
            dataset_sizes=("dataset_size", "nunique"),
        )
        .dropna(subset=["geom_speedup"])
    )
    include_baseline_in_combine = args.best == "smart-combine"
    per_size_df = compute_geom_by_dataset_size(
        speedups_for_plots,
        best=args.best,
        include_baseline=include_baseline_in_combine,
    )
    combined_geom_df = geom_mean_of_best_speedups(
        speedups_for_plots,
        include_baseline=include_baseline_in_combine,
        label="smart-combine" if include_baseline_in_combine else "combined",
    )
    tree_order = collect_polybench_algorithm_order(polybench_root)
    tree_categories = collect_polybench_categories(polybench_root)
    all_algorithms = gather_all_algorithms(translation_times, optimization_times_raw, outcomes)

    heatmap_table, heatmap_rows, heatmap_cols, heatmap_algs, heatmap_row_fws = build_heatmap_table(
        geom_df=geom_df,
        combined_geom_df=combined_geom_df,
        best=args.best,
        ordering=args.ordering,
        tree_order=tree_order,
        all_algorithms=all_algorithms,
        count_wins=args.count_wins,
        include_standard=args.standard_heat,
    )
    if not heatmap_table.empty:
        heatmap_stem_base = f"heatmap_speedups{best_suffix(args.best)}"
        heatmap_stem = append_label_suffix(heatmap_stem_base, label_suffix)
        heatmap_csv = out_dir / f"{heatmap_stem}.csv"
        heatmap_table.to_csv(heatmap_csv)
        print(f"Wrote heatmap table to {heatmap_csv}")
        heatmap_plot = out_dir / f"{heatmap_stem}.{plot_format}"
        plot_heatmap_table(
            heatmap_table,
            heatmap_rows,
            heatmap_cols,
            out_path=heatmap_plot,
            ordering=args.ordering,
            tree_categories=tree_categories,
            algorithms=heatmap_algs,
            row_frameworks=heatmap_row_fws,
            special_rows=["geom_mean"],
            special_cols=[col for col in heatmap_cols if col in ("geom_mean", "wins")],
            hide_colorbar=args.hide_colorbar,
        )
        print(f"Wrote heatmap plot to {heatmap_plot}")
    else:
        print("No data available to plot heatmap.")

    correctness_source = outcomes if args.standard_heat else outcomes_for_shares
    correctness_table, correctness_rows, correctness_cols, correctness_algs, correctness_row_fws = (
        build_correctness_heatmap_table(
            correctness_source,
            ordering=args.ordering,
            tree_order=tree_order,
            all_algorithms=all_algorithms,
            include_standard=args.standard_heat,
        )
    )
    if not correctness_table.empty:
        correctness_stem_base = f"heatmap_correctness{best_suffix(args.best)}"
        correctness_stem = append_label_suffix(correctness_stem_base, label_suffix)
        correctness_csv = out_dir / f"{correctness_stem}.csv"
        correctness_table.to_csv(correctness_csv)
        print(f"Wrote correctness heatmap table to {correctness_csv}")
        correctness_plot = out_dir / f"{correctness_stem}.{plot_format}"
        plot_heatmap_table(
            correctness_table,
            correctness_rows,
            correctness_cols,
            out_path=correctness_plot,
            ordering=args.ordering,
            tree_categories=tree_categories,
            algorithms=correctness_algs,
            row_frameworks=correctness_row_fws,
            cbar_label="Valid repetitions (%)",
            value_fmt="{:.0f}",
            value_fmt_overrides={"mean_valid": "{:.0f}"},
            cmap_name="Greens",
            special_rows=["mean_valid"],
            special_cols=[col for col in correctness_cols if col == "mean_valid"],
            hide_colorbar=args.hide_colorbar,
        )
        print(f"Wrote correctness heatmap plot to {correctness_plot}")
    else:
        print("No data available to plot correctness heatmap.")

    plot_per_framework_version(
        geom_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        combined_geom_df=combined_geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_per_framework_version_agg_algorithms(
        per_size_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        out_dir=out_dir,
        label_suffix=label_suffix,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_per_version_across_frameworks(
        geom_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        combined_geom_df=combined_geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_per_version_across_frameworks_agg_algorithms(
        per_size_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        out_dir=out_dir,
        label_suffix=label_suffix,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_per_framework_across_versions(
        geom_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        combined_geom_df=combined_geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_per_framework_across_versions_agg_algorithms(
        per_size_df,
        plot_kind=speedup_plot_kind,
        best=args.best,
        out_dir=out_dir,
        label_suffix=label_suffix,
        y_min=args.y_min,
        y_max=args.y_max,
        log_scale=args.log_scale,
        plot_format=plot_format,
    )
    plot_shares_per_framework_version(
        shares_fw_ver,
        geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        threshold=args.speedup_threshold,
        plot_format=plot_format,
    )
    plot_shares_per_version(
        shares_ver,
        geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        threshold=args.speedup_threshold,
        plot_format=plot_format,
    )
    plot_shares_per_framework(
        shares_fw,
        geom_df,
        out_dir=out_dir,
        label_suffix=label_suffix,
        ordering=args.ordering,
        tree_order=tree_order,
        tree_categories=tree_categories,
        all_algorithms=all_algorithms,
        threshold=args.speedup_threshold,
        plot_format=plot_format,
    )


if __name__ == "__main__":
    main()
