#!/usr/bin/env python3
"""Summarize attempt validity and exact best-of-k optimization outcomes.

An attempt is one (framework, prompt, benchmark, repetition) tuple.  It is
validated only when every selected dataset size is valid.  Attempt speedup is
the geometric mean across those dataset sizes, relative to the selected
standard baseline framework. The expected-speedup fallback remains the standard
C implementation.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator

from analyze_results import (
    BASELINE_FRAMEWORK,
    TUNING_BENCHMARKS,
    apply_validation_filter,
    collect_prefixed_files,
    collect_status_tokens,
    compute_speedups,
    filter_algorithms,
    filter_data_types,
    filter_dataset_sizes,
    filter_frameworks,
    geometric_mean,
    load_and_aggregate_times,
    load_and_aggregate_validation,
    merge_baseline_tables,
    normalize_cli_args,
    normalize_baseline_spec,
    normalize_framework_value,
    normalize_version_column,
    select_best_standard_baseline,
    select_standard_baseline,
)


EXPECTED_ATTEMPTS = 5
NUMERICAL_MISMATCH_STATUSES = {"false", "invalid", "validity_error"}
BEST_OF_K_FONT_SIZE = 18


def validate_five_attempts(validation: pd.DataFrame) -> None:
    """Require repetitions 1..5 for every framework/prompt/benchmark."""

    expected = set(range(1, EXPECTED_ATTEMPTS + 1))
    incomplete: list[str] = []
    for key, group in validation.groupby(["data_type", "framework", "version", "algorithm"]):
        actual = set(group["repetition"].unique())
        if actual != expected:
            incomplete.append(f"{key}: {sorted(actual)}")
    if incomplete:
        examples = "; ".join(incomplete[:5])
        raise ValueError(
            "Best-of-k requires exactly repetitions 1..5 for every group; "
            f"found {len(incomplete)} incomplete group(s), including {examples}"
        )


def build_attempt_table(
    validation: pd.DataFrame,
    speedups: pd.DataFrame,
    dataset_sizes: Sequence[str],
    c_fallback_speedups: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Collapse per-dataset validation and timing data into attempts."""

    expected_sizes = set(map(str, dataset_sizes))
    speedup_groups = {
        key: group
        for key, group in speedups.groupby(
            ["data_type", "framework", "version", "algorithm", "repetition"]
        )
    }
    records: list[dict[str, object]] = []
    keys = ["data_type", "framework", "version", "algorithm", "repetition"]

    for key, group in validation.groupby(keys):
        data_type, framework, prompt, benchmark, repetition = key
        present_sizes = set(group["dataset_size"].astype(str))
        validated = present_sizes == expected_sizes and bool(group["valid"].all())
        statuses = collect_status_tokens(group["status"])
        speedup_group = speedup_groups.get(key)
        has_complete_speedup = (
            speedup_group is not None
            and set(speedup_group["dataset_size"].astype(str)) == expected_sizes
        )
        attempt_speedup = (
            geometric_mean(speedup_group["speedup"])
            if validated and has_complete_speedup
            else math.nan
        )
        baseline_framework = (
            "|".join(sorted(speedup_group["baseline_framework"].astype(str).unique()))
            if has_complete_speedup
            else ""
        )
        fallback_key = (str(data_type), str(benchmark))
        if fallback_key not in c_fallback_speedups:
            raise RuntimeError(
                "Missing standard C fallback speedup for "
                f"{data_type}/{benchmark}"
            )
        c_fallback_speedup = c_fallback_speedups[fallback_key]
        failure_flags = {
            "compile_error": "compile_error" in statuses or "illegal" in statuses,
            "runtime_error": "runtime_error" in statuses,
            "numerical_mismatch": bool(statuses & NUMERICAL_MISMATCH_STATUSES),
            "timeout": "timeout" in statuses,
        }
        failure_reasons = "|".join(
            name for name, present in failure_flags.items() if present
        )
        records.append(
            {
                "data_type": data_type,
                "framework": framework,
                "prompt": prompt,
                "benchmark": benchmark,
                "repetition": int(repetition),
                "validated": validated,
                "performance_eligible": validated and np.isfinite(attempt_speedup),
                "attempt_speedup": attempt_speedup,
                "baseline_framework": baseline_framework,
                "c_fallback_speedup": c_fallback_speedup,
                "validated_dataset_count": int(group["valid"].sum()),
                "expected_dataset_count": len(expected_sizes),
                "failure_reasons": failure_reasons,
                **failure_flags,
                "unclassified_invalid": not validated and not any(failure_flags.values()),
            }
        )

    return pd.DataFrame.from_records(records)


def quantile_or_nan(values: pd.Series, quantile: float) -> float:
    """Return a quantile or NaN for an empty series."""

    return float(values.quantile(quantile)) if not values.empty else math.nan


def compute_failure_counts(attempts: pd.DataFrame) -> pd.DataFrame:
    """Count attempt-level failures; reason columns intentionally may overlap."""

    rows: list[dict[str, object]] = []
    for (data_type, framework, prompt), group in attempts.groupby(["data_type", "framework", "prompt"]):
        rows.append(
            {
                "data_type": data_type,
                "framework": framework,
                "prompt": prompt,
                "attempt_count": len(group),
                "validated_attempt_count": int(group["validated"].sum()),
                "invalid_attempt_count": int((~group["validated"]).sum()),
                "compile_error_count": int(group["compile_error"].sum()),
                "runtime_error_count": int(group["runtime_error"].sum()),
                "numerical_mismatch_count": int(group["numerical_mismatch"].sum()),
                "timeout_count": int(group["timeout"].sum()),
                "unclassified_invalid_count": int(group["unclassified_invalid"].sum()),
                "benchmark_count": group["benchmark"].nunique(),
            }
        )
    return pd.DataFrame.from_records(rows)


def enumerate_best_of_k(attempts: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all repetition subsets for k=1..5 for every benchmark."""

    records: list[dict[str, object]] = []
    group_keys = ["data_type", "framework", "prompt", "benchmark"]
    for (data_type, framework, prompt, benchmark), group in attempts.groupby(group_keys):
        ordered = group.sort_values("repetition").reset_index(drop=True)
        if len(ordered) != EXPECTED_ATTEMPTS:
            raise ValueError(
                f"Expected {EXPECTED_ATTEMPTS} attempts for "
                f"{framework}/{prompt}/{benchmark}, got {len(ordered)}"
            )
        for k in range(1, EXPECTED_ATTEMPTS + 1):
            for indices in itertools.combinations(range(EXPECTED_ATTEMPTS), k):
                subset = ordered.iloc[list(indices)]
                validated = subset[subset["validated"]]
                eligible = subset[subset["performance_eligible"]]
                best_speedup = (
                    float(eligible["attempt_speedup"].max())
                    if not eligible.empty
                    else math.nan
                )
                records.append(
                    {
                        "data_type": data_type,
                        "framework": framework,
                        "prompt": prompt,
                        "benchmark": benchmark,
                        "k": k,
                        "subset": "|".join(
                            str(value) for value in subset["repetition"]
                        ),
                        "has_validated_candidate": not validated.empty,
                        "conditional_best_speedup": best_speedup,
                        "baseline_framework": "|".join(
                            sorted(
                                value
                                for value in ordered["baseline_framework"].astype(str).unique()
                                if value
                            )
                        ),
                        "c_fallback_speedup": float(
                            ordered["c_fallback_speedup"].iloc[0]
                        ),
                        # The standard C implementation remains available when no
                        # generated candidate validates or when every candidate is slower.
                        "best_speedup_with_c_fallback": (
                            max(float(ordered["c_fallback_speedup"].iloc[0]), best_speedup)
                            if np.isfinite(best_speedup)
                            else float(ordered["c_fallback_speedup"].iloc[0])
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_best_of_k(subsets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact subset outcomes by framework, prompt, and budget."""

    rows: list[dict[str, object]] = []
    for (data_type, framework, prompt, k), group in subsets.groupby(
        ["data_type", "framework", "prompt", "k"]
    ):
        successful = group[group["has_validated_candidate"]]
        conditional = successful["conditional_best_speedup"].dropna()
        fallback_speedups = group["best_speedup_with_c_fallback"]
        expected_speedup = geometric_mean(fallback_speedups)
        baseline_framework = "|".join(
            sorted(
                value
                for value in group["baseline_framework"].astype(str).unique()
                if value
            )
        )
        rows.append(
            {
                "data_type": data_type,
                "framework": framework,
                "prompt": prompt,
                "k": int(k),
                "benchmark_count": group["benchmark"].nunique(),
                "subsets_per_benchmark": math.comb(EXPECTED_ATTEMPTS, int(k)),
                "subset_evaluation_count": len(group),
                "successful_subset_count": len(successful),
                "probability_validated_candidate": float(
                    group["has_validated_candidate"].mean()
                ),
                "conditional_best_speedup": geometric_mean(conditional),
                "conditional_best_speedup_median": (
                    float(conditional.median()) if not conditional.empty else math.nan
                ),
                "conditional_best_speedup_q1": quantile_or_nan(conditional, 0.25),
                "conditional_best_speedup_q3": quantile_or_nan(conditional, 0.75),
                "conditional_speedup_sample_count": len(conditional),
                "expected_speedup_with_c_fallback": expected_speedup,
                "expected_improvement_over_c_pct": (expected_speedup - 1.0) * 100.0,
                "baseline_framework": baseline_framework,
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_framework_prompt(
    attempts: pd.DataFrame, budget: pd.DataFrame
) -> pd.DataFrame:
    """Build the requested coverage and performance summary table."""

    budget_at_5 = budget[budget["k"] == EXPECTED_ATTEMPTS].set_index(
        ["data_type", "framework", "prompt"]
    )
    speedup_at_5 = budget_at_5["conditional_best_speedup"].to_dict()
    rows: list[dict[str, object]] = []
    for (data_type, framework, prompt), group in attempts.groupby(["data_type", "framework", "prompt"]):
        eligible = group[group["performance_eligible"]]
        values = eligible["attempt_speedup"].dropna()
        q1 = quantile_or_nan(values, 0.25)
        q3 = quantile_or_nan(values, 0.75)
        rows.append(
            {
                "data_type": data_type,
                "framework": framework,
                "prompt": prompt,
                "attempt_count": len(group),
                "validation_count": int(group["validated"].sum()),
                "validation_rate": float(group["validated"].mean()),
                "validation_rate_pct": float(group["validated"].mean() * 100.0),
                "performance_attempt_count": len(values),
                "median_speedup": (
                    float(values.median()) if not values.empty else math.nan
                ),
                "geometric_mean_speedup": geometric_mean(values),
                "speedup_q1": q1,
                "speedup_q3": q3,
                "speedup_iqr": q3 - q1,
                "speedup_at_5": speedup_at_5.get((data_type, framework, prompt), math.nan),
                "contributing_benchmark_count": eligible["benchmark"].nunique(),
                "benchmark_count": group["benchmark"].nunique(),
                "baseline_framework": "|".join(
                    sorted(
                        value
                        for value in group["baseline_framework"].astype(str).unique()
                        if value
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_budget_curves(budget: pd.DataFrame, out_path: Path) -> None:
    """Plot success probability and expected speedup as budget increases."""

    with plt.rc_context({"font.size": BEST_OF_K_FONT_SIZE}):
        _plot_budget_curves(budget, out_path)


def _plot_budget_curves(budget: pd.DataFrame, out_path: Path) -> None:
    """Render budget curves with the caller's active Matplotlib style."""

    prompts = sorted(budget["prompt"].unique())
    include_data_type = "data_type" in budget.columns and budget["data_type"].nunique(dropna=False) > 1
    series_cols = ["data_type", "framework"] if include_data_type else ["framework"]
    series_keys = list(budget[series_cols].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        2,
        len(prompts),
        figsize=(max(9.0, 3.2 * len(prompts)), 6.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")
    series_colors = {key: colors(i) for i, key in enumerate(series_keys)}
    marker_shapes = ["o", "s", "^", "D", "v", "P", "X"]
    series_markers = {
        key: marker_shapes[i % len(marker_shapes)] for i, key in enumerate(series_keys)
    }

    for column, prompt in enumerate(prompts):
        prompt_data = budget[budget["prompt"] == prompt]
        for key in series_keys:
            if include_data_type:
                data_type, framework = key
                data = prompt_data[
                    (prompt_data["data_type"] == data_type)
                    & (prompt_data["framework"] == framework)
                ].sort_values("k")
                label = f"{data_type} {normalize_framework_value(framework)}"
            else:
                (framework,) = key
                data = prompt_data[prompt_data["framework"] == framework].sort_values("k")
                label = normalize_framework_value(framework)
            if data.empty:
                continue
            axes[0, column].plot(
                data["k"],
                data["expected_speedup_with_c_fallback"],
                marker=series_markers[key],
                color=series_colors[key],
                label=label,
            )
            axes[1, column].plot(
                data["k"],
                data["probability_validated_candidate"] * 100.0,
                marker=series_markers[key],
                color=series_colors[key],
                label=label,
            )
        axes[0, column].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axes[0, column].set_title(prompt)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xticks(range(1, EXPECTED_ATTEMPTS + 1))
        axes[1, column].set_xlabel("Attempt budget k")

    axes[0, 0].set_ylabel("Expected speedup\n(with C fallback)")
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1, 0].set_ylabel("P(validated\ncandidate) (%)")
    minimum_probability = 100.0 * budget["probability_validated_candidate"].min()
    probability_bottom = max(
        0.0, 10.0 * math.floor((minimum_probability - 1.0) / 10.0)
    )
    axes[1, 0].set_ylim(bottom=probability_bottom)
    probability_ticks = sorted(
        {probability_bottom, *np.arange(0.0, 101.0, 20.0)}
    )
    probability_ticks = [
        tick for tick in probability_ticks if tick >= probability_bottom
    ]
    axes[1, 0].yaxis.set_major_locator(FixedLocator(probability_ticks))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(series_keys)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_c_fallback_speedups(
    baseline_table: pd.DataFrame, c_baseline: pd.DataFrame
) -> dict[tuple[str, str], float]:
    """Return selected-baseline-relative speedup for falling back to standard C."""

    if baseline_table.empty or c_baseline.empty:
        return {}

    selected = baseline_table[
        ["data_type", "algorithm", "dataset_size", "baseline_time_s"]
    ].rename(columns={"baseline_time_s": "selected_baseline_time_s"})
    c_rows = c_baseline[
        ["data_type", "algorithm", "dataset_size", "baseline_time_s"]
    ].rename(columns={"baseline_time_s": "c_time_s"})
    merged = selected.merge(c_rows, on=["data_type", "algorithm", "dataset_size"], how="inner")
    if merged.empty:
        return {}

    merged["c_fallback_speedup"] = merged["selected_baseline_time_s"] / merged["c_time_s"]
    merged = merged[np.isfinite(merged["c_fallback_speedup"]) & (merged["c_fallback_speedup"] > 0)]

    fallback: dict[tuple[str, str], float] = {}
    for (data_type, algorithm), group in merged.groupby(["data_type", "algorithm"]):
        fallback[(str(data_type), str(algorithm))] = geometric_mean(
            group["c_fallback_speedup"]
        )
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("translation_path")
    parser.add_argument("optimization_path")
    parser.add_argument(
        "--baseline",
        default=BASELINE_FRAMEWORK,
        help=(
            "Framework to use as the standard baseline. Per algorithm, the best "
            "standard repetition covering all dataset sizes is chosen; if unavailable, "
            "falls back to 'c' for that algorithm."
        ),
    )
    parser.add_argument("--output-dir", default="results/plots/attempt_analysis")
    parser.add_argument("--dataset-size", dest="dataset_sizes", action="append")
    parser.add_argument("--data-type", dest="data_types", action="append")
    parser.add_argument("--no-framework", action="append")
    parser.add_argument(
        "--include-tuning-benchmarks",
        action="store_true",
        help=(
            "Include prompt-tuning benchmarks in outputs. By default, 2mm, gemm, "
            "floyd-warshall, and heat-3d are excluded."
        ),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.baseline = normalize_baseline_spec(args.baseline)
    translation_root = Path(args.translation_path)
    optimization_root = Path(args.optimization_path)
    output_dir = Path(args.output_dir)

    translation_times = load_and_aggregate_times(
        collect_prefixed_files(translation_root, "times"), warmup=args.warmup
    )
    optimization_times = load_and_aggregate_times(
        collect_prefixed_files(optimization_root, "times"), warmup=args.warmup
    )
    translation_validation = load_and_aggregate_validation(
        collect_prefixed_files(translation_root, "validation")
    )
    optimization_validation = load_and_aggregate_validation(
        collect_prefixed_files(optimization_root, "validation")
    )
    if translation_validation is None or optimization_validation is None:
        raise RuntimeError("Both translation and optimization validation CSVs are required")

    translation_times = normalize_version_column(translation_times)
    optimization_times = normalize_version_column(optimization_times)
    translation_validation = normalize_version_column(translation_validation)
    optimization_validation = normalize_version_column(optimization_validation)

    dataset_filter = normalize_cli_args(args.dataset_sizes)
    data_type_filter = normalize_cli_args(args.data_types)
    excluded_frameworks = normalize_cli_args(args.no_framework)
    excluded_algorithms = (
        [] if args.include_tuning_benchmarks else list(TUNING_BENCHMARKS)
    )
    if dataset_filter:
        translation_times = filter_dataset_sizes(translation_times, dataset_filter)
        optimization_times = filter_dataset_sizes(optimization_times, dataset_filter)
        translation_validation = filter_dataset_sizes(
            translation_validation, dataset_filter
        )
        optimization_validation = filter_dataset_sizes(
            optimization_validation, dataset_filter
        )
    if data_type_filter:
        translation_times = filter_data_types(translation_times, data_type_filter)
        optimization_times = filter_data_types(optimization_times, data_type_filter)
        translation_validation = filter_data_types(translation_validation, data_type_filter)
        optimization_validation = filter_data_types(optimization_validation, data_type_filter)
    if excluded_frameworks:
        optimization_times = filter_frameworks(
            optimization_times, excluded_frameworks
        )
        optimization_validation = filter_frameworks(
            optimization_validation, excluded_frameworks
        )
    if excluded_algorithms:
        translation_times = filter_algorithms(translation_times, excluded_algorithms)
        optimization_times = filter_algorithms(optimization_times, excluded_algorithms)
        translation_validation = filter_algorithms(
            translation_validation, excluded_algorithms
        )
        optimization_validation = filter_algorithms(
            optimization_validation, excluded_algorithms
        )

    selected_sizes = sorted(
        set(optimization_validation["dataset_size"].astype(str))
        & set(translation_times["dataset_size"].astype(str))
    )
    if not selected_sizes:
        raise RuntimeError("No dataset sizes overlap between optimization and baseline data")

    validate_five_attempts(optimization_validation)
    valid_translation_times = apply_validation_filter(
        translation_times, translation_validation
    )

    primary_baseline = select_standard_baseline(
        valid_translation_times,
        baseline=args.baseline,
        dataset_sizes=selected_sizes,
    )
    c_baseline = select_best_standard_baseline(
        valid_translation_times,
        baseline_framework=BASELINE_FRAMEWORK,
        dataset_sizes=selected_sizes,
    )
    fallback_baseline = (
        c_baseline
        if args.baseline != BASELINE_FRAMEWORK
        else None
    )
    baseline_table = merge_baseline_tables(primary_baseline, fallback_baseline)
    if baseline_table.empty:
        raise RuntimeError(
            "No complete standard baseline is available for the requested framework(s)"
        )
    if c_baseline.empty:
        raise RuntimeError("No complete standard C baseline is available for fallback")
    c_fallback_speedups = compute_c_fallback_speedups(baseline_table, c_baseline)

    valid_optimization_times = apply_validation_filter(
        optimization_times, optimization_validation
    )
    speedups = compute_speedups(
        valid_translation_times,
        valid_optimization_times,
        baseline_table=baseline_table,
    )

    attempts = build_attempt_table(
        optimization_validation, speedups, selected_sizes, c_fallback_speedups
    )
    subsets = enumerate_best_of_k(attempts)
    budget = summarize_best_of_k(subsets)
    failures = compute_failure_counts(attempts)
    budget = budget.merge(
        failures.drop(columns=["benchmark_count"]),
        on=["data_type", "framework", "prompt"],
        how="left",
    )
    summary = summarize_framework_prompt(attempts, budget)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "framework_prompt_summary.csv"
    failures_path = output_dir / "failure_counts.csv"
    budget_path = output_dir / "best_of_k.csv"
    plot_path = output_dir / f"best_of_k_budget.{args.format}"
    summary.sort_values(["data_type", "framework", "prompt"]).to_csv(summary_path, index=False)
    failures.sort_values(["data_type", "framework", "prompt"]).to_csv(failures_path, index=False)
    budget.sort_values(["data_type", "framework", "prompt", "k"]).to_csv(budget_path, index=False)
    plot_budget_curves(budget, plot_path)

    print(f"Wrote framework/prompt summary to {summary_path}")
    print(f"Wrote failure counts to {failures_path}")
    print(f"Wrote exact best-of-k table to {budget_path}")
    print(f"Wrote budget visualization to {plot_path}")


if __name__ == "__main__":
    main()
