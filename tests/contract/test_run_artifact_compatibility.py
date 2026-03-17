from __future__ import annotations

import json
from pathlib import Path

import pytest

from label_master.reports.schemas import (
    CURRENT_SCHEMA_VERSION,
    parse_run_config,
    parse_run_report,
)

FIXTURES = Path("tests/fixtures/contracts")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    ["run_config.valid.v1_1.json", "run_config.valid.v1_0.json"],
)
def test_run_config_compatibility_current_and_previous_minor(fixture_name: str) -> None:
    model = parse_run_config(_load(FIXTURES / fixture_name))
    assert model.schema_version == CURRENT_SCHEMA_VERSION


@pytest.mark.parametrize(
    "fixture_name",
    ["run_report.valid.v1_1.json", "run_report.valid.v1_0.json"],
)
def test_run_report_compatibility_current_and_previous_minor(fixture_name: str) -> None:
    model = parse_run_report(_load(FIXTURES / fixture_name))
    assert model.schema_version == CURRENT_SCHEMA_VERSION
