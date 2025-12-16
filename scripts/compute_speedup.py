#!/usr/bin/env python3

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compute_speedup.py <baseline_seconds> <optimized_seconds>", file=sys.stderr)
        return 1

    baseline_str, optimized_str = sys.argv[1:3]

    try:
        baseline = float(baseline_str)
        optimized = float(optimized_str)
    except ValueError as exc:
        print(f"Invalid runtime value: {exc}", file=sys.stderr)
        return 1

    if optimized <= 0:
        print("Optimized runtime must be positive.", file=sys.stderr)
        return 1

    print(f"{baseline:.6f} {optimized:.6f} {baseline / optimized:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
