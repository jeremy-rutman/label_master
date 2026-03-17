from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
FIXTURES = Path("tests/fixtures/us1")


def test_yolo_to_coco_ordering_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    args = [
        "convert",
        "--input",
        str(FIXTURES / "yolo_minimal"),
        "--src",
        "yolo",
        "--dst",
        "coco",
    ]

    result_a = RUNNER.invoke(app, [*args, "--output", str(out_a)])
    result_b = RUNNER.invoke(app, [*args, "--output", str(out_b)])

    assert result_a.exit_code == 0
    assert result_b.exit_code == 0

    content_a = (out_a / "annotations.json").read_text(encoding="utf-8")
    content_b = (out_b / "annotations.json").read_text(encoding="utf-8")
    assert content_a == content_b
