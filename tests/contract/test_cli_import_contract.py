from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from label_master.interfaces.cli.main import app

RUNNER = CliRunner()


def test_cli_import_rejects_unknown_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = RUNNER.invoke(
        app,
        [
            "import",
            "--provider",
            "unknown",
            "--source-ref",
            str(Path("tests/fixtures/us3/provider_sample")),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 5


def test_cli_import_accepts_provider_arguments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = RUNNER.invoke(
        app,
        [
            "import",
            "--provider",
            "kaggle",
            "--source-ref",
            str(Path("tests/fixtures/us3/provider_sample")),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
