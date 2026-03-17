#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that review evidence links specs and tests")
    parser.add_argument("--evidence", type=Path, required=True, help="Path to markdown/text evidence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.evidence.exists():
        print(f"evidence file not found: {args.evidence}", file=sys.stderr)
        return 2

    text = args.evidence.read_text(encoding="utf-8")

    has_spec_ref = bool(re.search(r"specs/[\w\-]+/", text))
    has_test_ref = bool(re.search(r"tests/[\w\-/\.]+", text))

    if not has_spec_ref or not has_test_ref:
        print(
            "traceability check failed: evidence must reference both a spec path and a test path",
            file=sys.stderr,
        )
        return 1

    print("traceability check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
