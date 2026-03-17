from __future__ import annotations

from label_master.core.services.import_service import import_dataset


def test_direct_url_file_protocol_warning_recorded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "annotations.json").write_text("{}", encoding="utf-8")

    result = import_dataset(
        provider="direct_url",
        source_ref=f"file://{source_dir}",
        output_path=tmp_path / "out",
        run_id="job-direct-file",
    )

    warnings = result.report.warnings
    assert warnings
    assert warnings[0].severity == "warning"
    assert result.report.provenance[0].protocol == "file"
