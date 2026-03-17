#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce regression-test evidence for bug fixes")
    parser.add_argument("--evidence", type=Path, required=True, help="Path to review evidence file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.evidence.exists():
        print(f"evidence file not found: {args.evidence}", file=sys.stderr)
        return 2

    text = args.evidence.read_text(encoding="utf-8")

    bugfix_declared = bool(re.search(r"(?im)^\s*bugfix\s*:\s*true\s*$", text))
    if not bugfix_declared:
        print("regression gate passed (not marked as bugfix)")
        return 0

    regression_match = re.search(r"(?im)^\s*regression-test\s*:\s*(tests/[\w\-/\.]+)\s*$", text)
    if not regression_match:
        print("regression gate failed: bugfix requires 'Regression-Test: tests/...'", file=sys.stderr)
        return 1

    test_path = Path(regression_match.group(1))
    if not test_path.exists():
        print(f"regression gate failed: referenced test does not exist: {test_path}", file=sys.stderr)
        return 1

    print("regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
