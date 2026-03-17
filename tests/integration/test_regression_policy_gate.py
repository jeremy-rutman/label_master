from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/ci/check_regression_tests.py")


def test_regression_gate_passes_when_bugfix_has_regression_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    evidence = tmp_path / "evidence.md"
    evidence.write_text(
        "Bugfix: true\nRegression-Test: tests/integration/test_regression_policy_gate.py\n",
        encoding="utf-8",
    )

    completed = subprocess.run([sys.executable, str(SCRIPT), "--evidence", str(evidence)], check=False)
    assert completed.returncode == 0


def test_regression_gate_fails_when_bugfix_missing_regression_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    evidence = tmp_path / "evidence.md"
    evidence.write_text("Bugfix: true\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(SCRIPT), "--evidence", str(evidence)], check=False)
    assert completed.returncode == 1
