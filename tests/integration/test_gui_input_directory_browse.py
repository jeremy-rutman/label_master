from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui import app
from label_master.interfaces.gui.system_actions import DirectoryBrowseResult


def test_input_directory_browse_success_and_cancel(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    selected = tmp_path / "selected"
    selected.mkdir()

    monkeypatch.setattr(
        app.system_actions,
        "browse_for_directory",
        lambda *, initial_directory=None: DirectoryBrowseResult(
            selected_path=selected,
            available=True,
            message="Selected directory.",
        ),
    )
    success = app.attempt_input_directory_browse("tests/fixtures/us1/coco_minimal")
    assert success.input_dir_raw == str(selected.resolve())
    assert success.browse_available is True
    assert success.browse_message == "Selected directory."

    monkeypatch.setattr(
        app.system_actions,
        "browse_for_directory",
        lambda *, initial_directory=None: DirectoryBrowseResult(
            selected_path=None,
            available=True,
            message="Directory selection cancelled.",
        ),
    )
    cancelled = app.attempt_input_directory_browse(success.input_dir_raw)
    assert cancelled.input_dir_raw == success.input_dir_raw
    assert cancelled.browse_available is True
    assert cancelled.browse_message is not None
    assert "cancel" in cancelled.browse_message.lower()


def test_input_directory_browse_unavailable_fallback_keeps_manual_entry(
    monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    manual_path = Path(tmp_path).resolve()

    monkeypatch.setattr(
        app.system_actions,
        "browse_for_directory",
        lambda *, initial_directory=None: DirectoryBrowseResult(
            selected_path=None,
            available=False,
            message="Browse unavailable; enter path manually.",
        ),
    )
    browsed = app.attempt_input_directory_browse("tests/fixtures/us1/coco_minimal")
    assert browsed.input_dir_raw == "tests/fixtures/us1/coco_minimal"
    assert browsed.browse_available is False
    assert browsed.browse_message == "Browse unavailable; enter path manually."

    validation = app.validate_input_directory(str(manual_path))
    assert validation.errors == []
