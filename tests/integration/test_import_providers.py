from __future__ import annotations

from pathlib import Path

import pytest

from label_master.core.services.import_service import import_dataset

SOURCE = Path("tests/fixtures/us3/provider_sample")


def test_import_kaggle_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = import_dataset(
        provider="kaggle",
        source_ref=str(SOURCE),
        output_path=tmp_path / "kaggle",
        run_id="job-kaggle",
    )
    assert result.local_path.exists()


def test_import_roboflow_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = import_dataset(
        provider="roboflow",
        source_ref=str(SOURCE),
        output_path=tmp_path / "roboflow",
        run_id="job-roboflow",
    )
    assert result.local_path.exists()


def test_import_github_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = import_dataset(
        provider="github",
        source_ref=str(SOURCE),
        output_path=tmp_path / "github",
        run_id="job-github",
    )
    assert result.local_path.exists()


def test_import_direct_url_file_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    local_archive_dir = tmp_path / "input_dir"
    local_archive_dir.mkdir()
    (local_archive_dir / "annotations.json").write_text("{}", encoding="utf-8")

    source_ref = f"file://{local_archive_dir}"
    result = import_dataset(
        provider="direct_url",
        source_ref=source_ref,
        output_path=tmp_path / "direct",
        run_id="job-direct-file",
    )
    assert result.local_path.exists()


def test_import_fails_closed_for_malformed_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from label_master.core.domain.value_objects import ImportError

    with pytest.raises(ImportError):
        import_dataset(
            provider="kaggle",
            source_ref=str(Path("tests/fixtures/us3/malformed_empty")),
            output_path=tmp_path / "bad",
            run_id="job-bad",
        )
