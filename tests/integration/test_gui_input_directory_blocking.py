from __future__ import annotations

from label_master.interfaces.gui.app import run_blocking_errors


def test_run_blocking_for_non_directory_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_file = tmp_path / "input.txt"
    input_file.write_text("not-a-directory", encoding="utf-8")

    errors = run_blocking_errors(
        input_dir_raw=str(input_file),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=[],
    )

    assert "Input directory must be a directory" in errors


def test_run_blocking_for_unreadable_directory(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setattr("label_master.interfaces.gui.app._is_readable_directory", lambda path: False)

    errors = run_blocking_errors(
        input_dir_raw=str(input_dir),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=[],
    )

    assert "Input directory must be readable" in errors
