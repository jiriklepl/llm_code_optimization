#!/usr/bin/env python3

import re
import sys


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: extract_runtime.py <stdout_file>", file=sys.stderr)
        return 1

    path = sys.argv[1]

    try:
        with open(path, encoding="utf-8") as handle:
            data = handle.read()
    except OSError as exc:
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        return 1

    matches = NUMBER_PATTERN.findall(data)
    if not matches:
        print(f"No numeric values found in {path}", file=sys.stderr)
        return 1

    print(matches[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
