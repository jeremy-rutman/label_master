from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "us1"


def test_cli_infer_contract() -> None:
    result = RUNNER.invoke(
        app,
        [
            "infer",
            "--input",
            str(FIXTURES / "coco_minimal"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["predicted_format"] == "coco"


def test_cli_validate_contract() -> None:
    result = RUNNER.invoke(
        app,
        [
            "validate",
            "--input",
            str(FIXTURES / "coco_minimal"),
            "--format",
            "coco",
            "--strict",
        ],
    )
    assert result.exit_code == 0
    assert "valid=True" in result.stdout


def test_cli_convert_contract_with_unmapped_error() -> None:
    with RUNNER.isolated_filesystem():
        output = Path("out")
        result = RUNNER.invoke(
            app,
            [
                "convert",
                "--input",
                str(FIXTURES / "coco_minimal"),
                "--output",
                str(output),
                "--src",
                "coco",
                "--dst",
                "yolo",
                "--unmapped-policy",
                "error",
            ],
        )

    assert result.exit_code == 0


def test_cli_remap_contract() -> None:
    with RUNNER.isolated_filesystem():
        output = Path("remap_out")
        result = RUNNER.invoke(
            app,
            [
                "remap",
                "--input",
                str(FIXTURES / "coco_minimal"),
                "--output",
                str(output),
                "--format",
                "coco",
                "--map",
                str(FIXTURES / "maps" / "class_map_drop.yaml"),
            ],
        )

    assert result.exit_code == 0
    assert "remap_complete" in result.stdout


def test_cli_infer_ambiguous_exit_code_3() -> None:
    with RUNNER.isolated_filesystem():
        mixed = Path("mixed")
        mixed.mkdir(parents=True)
        (mixed / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (mixed / "annotations.json").write_text(
            json.dumps({"images": [], "annotations": []}),
            encoding="utf-8",
        )

        result = RUNNER.invoke(app, ["infer", "--input", str(mixed)])

    assert result.exit_code == 3
