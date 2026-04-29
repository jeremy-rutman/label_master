from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectoryBrowseResult:
    selected_path: Path | None
    available: bool
    message: str | None


@dataclass(frozen=True)
class OutputDirectoryOpenResult:
    requested_path: Path
    opened: bool
    message: str


class DirectoryDialogUnavailableError(RuntimeError):
    """Raised when no usable directory dialog backend is available."""


def _normalized_dialog_title(dialog_title: str) -> str:
    normalized = dialog_title.strip()
    return normalized or "Select directory"


def _open_native_directory_dialog(
    *,
    initial_directory: Path | None = None,
    dialog_title: str = "Select directory",
) -> Path | None:
    if threading.current_thread() is not threading.main_thread():
        raise DirectoryDialogUnavailableError("tkinter directory picker is only safe from the main thread")

    import tkinter
    from tkinter import filedialog

    title = _normalized_dialog_title(dialog_title)
    root = tkinter.Tk()
    try:
        root.withdraw()
        selected = filedialog.askdirectory(
            initialdir=str(initial_directory) if initial_directory else None,
            mustexist=True,
            title=title,
        )
    finally:
        root.destroy()

    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def _run_dialog_command(command: list[str]) -> Path | None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode in {0}:
        selected = completed.stdout.strip()
        if not selected:
            return None
        return Path(selected).expanduser().resolve()
    if completed.returncode in {1, 130}:
        return None
    stderr = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
    raise RuntimeError(stderr)


def _open_fallback_directory_dialog(
    *,
    initial_directory: Path | None = None,
    dialog_title: str = "Select directory",
) -> Path | None:
    initial_arg = str(initial_directory) if initial_directory else str(Path.home())
    title = _normalized_dialog_title(dialog_title)
    errors: list[str] = []

    if sys.platform.startswith("darwin") and shutil.which("osascript"):
        escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
        script = f'POSIX path of (choose folder with prompt "{escaped_title}")'
        try:
            return _run_dialog_command(["osascript", "-e", script])
        except Exception as exc:
            errors.append(f"osascript: {exc}")

    if sys.platform.startswith("win"):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            escaped_initial = initial_arg.replace("'", "''")
            escaped_title = title.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f"$dialog.Description = '{escaped_title}'; "
                f"$dialog.SelectedPath = '{escaped_initial}'; "
                "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
                "{ Write-Output $dialog.SelectedPath; exit 0 } "
                "else { exit 1 }"
            )
            try:
                return _run_dialog_command([powershell, "-NoProfile", "-Command", script])
            except Exception as exc:
                errors.append(f"powershell: {exc}")

    if shutil.which("zenity"):
        try:
            return _run_dialog_command(
                [
                    "zenity",
                    "--file-selection",
                    "--directory",
                    f"--title={title}",
                    f"--filename={initial_arg}/",
                ]
            )
        except Exception as exc:
            errors.append(f"zenity: {exc}")

    if shutil.which("kdialog"):
        try:
            return _run_dialog_command(["kdialog", "--title", title, "--getexistingdirectory", initial_arg])
        except Exception as exc:
            errors.append(f"kdialog: {exc}")

    if shutil.which("yad"):
        try:
            return _run_dialog_command(
                [
                    "yad",
                    "--file",
                    "--directory",
                    f"--title={title}",
                    f"--filename={initial_arg}/",
                ]
            )
        except Exception as exc:
            errors.append(f"yad: {exc}")

    if errors:
        raise DirectoryDialogUnavailableError("; ".join(errors))
    raise DirectoryDialogUnavailableError("No supported directory picker backend found")


def browse_for_directory(
    *,
    initial_directory: Path | None = None,
    dialog_title: str = "Select directory",
) -> DirectoryBrowseResult:
    try:
        selected = _open_native_directory_dialog(
            initial_directory=initial_directory,
            dialog_title=dialog_title,
        )
    except Exception as native_exc:
        try:
            selected = _open_fallback_directory_dialog(
                initial_directory=initial_directory,
                dialog_title=dialog_title,
            )
        except DirectoryDialogUnavailableError as fallback_exc:  # pragma: no cover - exercised via monkeypatch
            return DirectoryBrowseResult(
                selected_path=None,
                available=False,
                message=f"Browse unavailable ({native_exc}; {fallback_exc}). Enter a path manually.",
            )

    if selected is None:
        return DirectoryBrowseResult(
            selected_path=None,
            available=True,
            message="Directory selection cancelled.",
        )

    return DirectoryBrowseResult(
        selected_path=selected,
        available=True,
        message="Selected directory.",
    )


def _open_path_with_platform_default(path: Path) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", str(path)], check=True)
        return

    if sys.platform.startswith("win"):
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise RuntimeError("os.startfile is unavailable on this platform")
        startfile(str(path))
        return

    subprocess.run(["xdg-open", str(path)], check=True)


def open_output_directory(path: Path) -> OutputDirectoryOpenResult:
    requested_path = path.expanduser().resolve()

    if not requested_path.exists():
        return OutputDirectoryOpenResult(
            requested_path=requested_path,
            opened=False,
            message=f"Output directory does not exist: {requested_path}",
        )

    if not requested_path.is_dir():
        return OutputDirectoryOpenResult(
            requested_path=requested_path,
            opened=False,
            message=f"Output path is not a directory: {requested_path}",
        )

    try:
        _open_path_with_platform_default(requested_path)
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        return OutputDirectoryOpenResult(
            requested_path=requested_path,
            opened=False,
            message=f"Could not open automatically ({exc}). Access manually: {requested_path}",
        )

    return OutputDirectoryOpenResult(
        requested_path=requested_path,
        opened=True,
        message=f"Opened output directory: {requested_path}",
    )
