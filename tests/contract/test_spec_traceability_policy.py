from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/ci/check_spec_traceability.py")


def test_spec_traceability_policy_passes_with_spec_and_test_refs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    evidence = tmp_path / "evidence.md"
    evidence.write_text(
        "Spec: specs/001-annotation-collab-ingestion/spec.md\n"
        "Test: tests/integration/test_cli_convert_remap.py\n",
        encoding="utf-8",
    )

    completed = subprocess.run([sys.executable, str(SCRIPT), "--evidence", str(evidence)], check=False)
    assert completed.returncode == 0


def test_spec_traceability_policy_fails_without_test_ref(tmp_path) -> None:  # type: ignore[no-untyped-def]
    evidence = tmp_path / "evidence.md"
    evidence.write_text("Spec: specs/001-annotation-collab-ingestion/spec.md\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(SCRIPT), "--evidence", str(evidence)], check=False)
    assert completed.returncode == 1
