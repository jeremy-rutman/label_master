from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui import system_actions


def test_browse_directory_unavailable_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _raise_dialog(*, initial_directory: Path | None = None) -> Path | None:
        raise RuntimeError("headless")

    def _raise_fallback(*, initial_directory: Path | None = None) -> Path | None:
        raise system_actions.DirectoryDialogUnavailableError("no dialog backend")

    monkeypatch.setattr(system_actions, "_open_native_directory_dialog", _raise_dialog)
    monkeypatch.setattr(system_actions, "_open_fallback_directory_dialog", _raise_fallback)

    result = system_actions.browse_for_directory()

    assert result.available is False
    assert result.selected_path is None
    assert result.message is not None
    assert "manual" in result.message.lower()


def test_browse_directory_cancel_keeps_available(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        system_actions,
        "_open_native_directory_dialog",
        lambda *, initial_directory=None: None,
    )

    result = system_actions.browse_for_directory()

    assert result.available is True
    assert result.selected_path is None
    assert result.message is not None
    assert "cancel" in result.message.lower()


def test_browse_directory_uses_fallback_when_tk_unavailable(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def _raise_dialog(*, initial_directory: Path | None = None) -> Path | None:
        raise ModuleNotFoundError("No module named 'tkinter'")

    monkeypatch.setattr(system_actions, "_open_native_directory_dialog", _raise_dialog)
    monkeypatch.setattr(
        system_actions,
        "_open_fallback_directory_dialog",
        lambda *, initial_directory=None: tmp_path.resolve(),
    )

    result = system_actions.browse_for_directory(initial_directory=tmp_path)

    assert result.available is True
    assert result.selected_path == tmp_path.resolve()
    assert result.message == "Selected directory."


def test_open_output_directory_success(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(system_actions, "_open_path_with_platform_default", lambda path: None)

    result = system_actions.open_output_directory(tmp_path)

    assert result.opened is True
    assert str(tmp_path.resolve()) in result.message


def test_open_output_directory_failure_returns_fallback(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def _raise_open(path: Path) -> None:
        raise RuntimeError("open failed")

    monkeypatch.setattr(system_actions, "_open_path_with_platform_default", _raise_open)

    result = system_actions.open_output_directory(tmp_path)

    assert result.opened is False
    assert "open failed" in result.message
    assert str(tmp_path.resolve()) in result.message


def test_open_output_directory_missing_path() -> None:
    missing = Path("/tmp/label_master_gui_missing_dir")
    result = system_actions.open_output_directory(missing)

    assert result.opened is False
    assert "does not exist" in result.message.lower()
    assert str(missing.resolve()) in result.message
