from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui import app
from label_master.interfaces.gui.system_actions import OutputDirectoryOpenResult


def test_output_directory_action_fallback_path_visible(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    output_dir = Path(tmp_path).resolve()

    monkeypatch.setattr(
        app.system_actions,
        "open_output_directory",
        lambda path: OutputDirectoryOpenResult(
            requested_path=path.resolve(),
            opened=False,
            message=f"Could not open automatically. Access manually: {path.resolve()}",
        ),
    )

    result = app.attempt_output_directory_access(str(output_dir))
    assert result.opened is False
    assert str(output_dir) in result.message
