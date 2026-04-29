from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "us1"


def test_coco_to_yolo_convert_with_remap_and_drop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"
    report_dir = tmp_path / "reports"

    result = RUNNER.invoke(
        app,
        [
            "--report-path",
            str(report_dir),
            "convert",
            "--input",
            str(FIXTURES / "coco_minimal"),
            "--output",
            str(output),
            "--src",
            "coco",
            "--dst",
            "yolo",
            "--map",
            str(FIXTURES / "maps" / "class_map_drop.yaml"),
            "--unmapped-policy",
            "drop",
        ],
    )

    assert result.exit_code == 0
    label_files = sorted((output / "labels").glob("*.txt"))
    assert len(label_files) == 2
    expected_stems = sorted(path.stem for path in (FIXTURES / "coco_minimal" / "images").iterdir())
    assert sorted(path.stem for path in label_files) == expected_stems

    class_ids = sorted(int(path.read_text(encoding="utf-8").split()[0]) for path in label_files)
    assert class_ids == [10, 11]

    reports = list(report_dir.glob("*.report.json"))
    assert reports
    warnings_logs = list(report_dir.glob("*.warnings.json"))
    assert warnings_logs == []
    dropped_logs = list(report_dir.glob("*.dropped_annotations.json"))
    assert len(dropped_logs) == 1
    dropped_payload = json.loads(dropped_logs[0].read_text(encoding="utf-8"))
    assert dropped_payload["dropped_annotation_count"] == 5
    assert all(item["stage"] == "remap" for item in dropped_payload["dropped_annotations"])
    assert all(item["reason_code"] == "class_dropped_by_mapping" for item in dropped_payload["dropped_annotations"])


def test_convert_dry_run_does_not_write_annotations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"
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
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert not (output / "labels").exists()
