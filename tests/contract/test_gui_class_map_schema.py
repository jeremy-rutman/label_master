from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from label_master.interfaces.gui.app import persist_generated_class_map

SCHEMA_PATH = Path("specs/002-enhance-gui/contracts/gui-class-map.schema.json")
FIXTURES = Path("tests/fixtures/contracts")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _load(SCHEMA_PATH)
    return Draft202012Validator(schema)


def test_gui_class_map_schema_accepts_valid_fixture() -> None:
    payload = _load(FIXTURES / "gui-class-map.valid.json")
    _validator().validate(payload)


def test_gui_class_map_schema_rejects_invalid_fixture() -> None:
    payload = _load(FIXTURES / "gui-class-map.invalid.json")
    with pytest.raises(ValidationError):
        _validator().validate(payload)


def test_gui_persisted_class_map_artifact_matches_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    artifact_path = persist_generated_class_map(
        {2: None, 3: 10},
        run_id="gui-schema-check",
        reports_dir=tmp_path,
    )
    payload = _load(artifact_path)
    _validator().validate(payload)
    assert payload == {"class_map": {"2": None, "3": 10}}
