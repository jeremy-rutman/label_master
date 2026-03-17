from __future__ import annotations

from label_master.interfaces.gui.app import run_blocking_errors, validate_input_directory


def test_validate_input_directory_requires_value() -> None:
    result = validate_input_directory("")
    assert result.errors == ["Input directory is required"]


def test_validate_input_directory_missing_path() -> None:
    result = validate_input_directory("/tmp/label_master_missing_path_for_gui_validation")
    assert result.errors == ["Input directory does not exist"]


def test_validate_input_directory_requires_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")

    result = validate_input_directory(str(file_path))
    assert result.errors == ["Input directory must be a directory"]


def test_validate_input_directory_requires_readable_directory(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("label_master.interfaces.gui.app._is_readable_directory", lambda path: False)

    result = validate_input_directory(str(tmp_path))
    assert result.errors == ["Input directory must be readable"]


def test_validate_input_directory_valid_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = validate_input_directory(str(tmp_path))
    assert result.errors == []
    assert result.resolved_path == tmp_path.resolve()


def test_run_blocking_errors_include_mapping_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    errors = run_blocking_errors(
        input_dir_raw=str(tmp_path),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=["Row 2: duplicate source_class_id 3"],
    )
    assert "Row 2: duplicate source_class_id 3" in errors
