from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from label_master.interfaces.cli.main import app
from label_master.interfaces.gui.viewmodels import convert_view

RUNNER = CliRunner()


def test_gui_cli_output_parity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")

    gui_output = tmp_path / "gui_output"
    cli_output = tmp_path / "cli_output"

    vm, _ = convert_view(
        input_path=source,
        output_path=gui_output,
        src="coco",
        dst="yolo",
        map_path=Path("tests/fixtures/us1/maps/class_map_drop.yaml"),
        unmapped_policy="drop",
        dry_run=False,
    )
    assert vm.annotations_out > 0

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
            "tests/fixtures/us1/maps/class_map_drop.yaml",
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
