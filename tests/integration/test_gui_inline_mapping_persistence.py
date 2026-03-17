from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from label_master.infra.config import load_mapping_file
from label_master.interfaces.cli.main import app
from label_master.interfaces.gui.app import persist_generated_class_map
from label_master.interfaces.gui.viewmodels import (
    MappingRowViewModel,
    convert_view,
    parse_mapping_rows,
)

RUNNER = CliRunner()


def test_gui_inline_mapping_persistence_and_cli_parity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="2", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="map", destination_class_id="10"),
            MappingRowViewModel(source_class_id="4", action="map", destination_class_id="11"),
        ]
    )
    assert parsed.errors == []

    reports_dir = tmp_path / "reports"
    map_path = persist_generated_class_map(
        parsed.class_map,
        run_id="gui-inline-test",
        reports_dir=reports_dir,
    )
    assert map_path == reports_dir / "gui-inline-test.gui.class_map.json"
    assert load_mapping_file(map_path) == parsed.class_map
    schema = json.loads(
        Path("specs/002-enhance-gui/contracts/gui-class-map.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(json.loads(map_path.read_text(encoding="utf-8")))

    source = Path("tests/fixtures/us1/coco_minimal")
    gui_output = tmp_path / "gui_output"
    cli_output = tmp_path / "cli_output"

    vm, _ = convert_view(
        input_path=source,
        output_path=gui_output,
        src="coco",
        dst="yolo",
        map_path=map_path,
        unmapped_policy="drop",
        dry_run=False,
    )
    assert vm.annotations_out > 0
    assert vm.annotations_in > vm.annotations_out
    assert vm.dropped > 0

    cli_result = RUNNER.invoke(
        app,
        [
            "convert",
            "--input",
            str(source),
            "--output",
            str(cli_output),
            "--src",
            "coco",
            "--dst",
            "yolo",
            "--map",
            str(map_path),
            "--unmapped-policy",
            "drop",
        ],
    )
    assert cli_result.exit_code == 0

    gui_labels = sorted((gui_output / "labels").rglob("*.txt"))
    cli_labels = sorted((cli_output / "labels").rglob("*.txt"))

    assert [p.name for p in gui_labels] == [p.name for p in cli_labels]
    assert [p.read_text(encoding="utf-8") for p in gui_labels] == [
        p.read_text(encoding="utf-8") for p in cli_labels
    ]
