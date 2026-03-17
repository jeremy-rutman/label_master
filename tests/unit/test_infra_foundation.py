from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from label_master.core.domain.value_objects import PathTraversalError
from label_master.infra.filesystem import (
    atomic_write_json,
    safe_file_uri_to_path,
    safe_resolve,
)
from label_master.infra.locking import OutputPathLockManager
from label_master.infra.reporting import generate_run_id, persist_run_artifacts
from label_master.reports.schemas import RunConfigModel, RunReportModel, SummaryCountsModel


def test_safe_resolve_blocks_path_escape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = tmp_path / "workspace"
    base.mkdir()

    safe = safe_resolve(base, "nested/file.txt")
    assert str(safe).startswith(str(base))

    with pytest.raises(PathTraversalError):
        safe_resolve(base, "../escape.txt")


def test_safe_file_uri_enforces_allowed_root(tmp_path) -> None:  # type: ignore[no-untyped-def]
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "example.zip"
    inside.write_text("x", encoding="utf-8")

    resolved = safe_file_uri_to_path(f"file://{inside}", allowed)
    assert resolved == inside.resolve()

    outside = tmp_path / "outside.zip"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(PathTraversalError):
        safe_file_uri_to_path(f"file://{outside}", allowed)


def test_lock_manager_records_contention(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = OutputPathLockManager(lock_root=tmp_path / "locks")
    output_path = tmp_path / "out"

    first = manager.acquire(output_path, "run-a")
    second = manager.acquire(output_path, "run-b")

    assert first == []
    assert len(second) == 1
    assert second[0].competing_run_id == "run-a"
    assert manager.get_owner(output_path) == "run-b"


def test_reporting_persists_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_id = generate_run_id()
    config = RunConfigModel(
        run_id=run_id,
        mode="infer",
        input_path="/tmp/in",
        src_format="auto",
        created_at=datetime.now(UTC),
    )
    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path="/tmp/in",
        summary_counts=SummaryCountsModel(
            images=1,
            annotations_in=1,
            annotations_out=1,
            dropped=0,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
    )

    config_path, report_path = persist_run_artifacts(tmp_path / "reports", run_id, config, report)
    assert config_path.exists()
    assert report_path.exists()

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id


def test_atomic_write_json_writes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "nested" / "payload.json"
    atomic_write_json(target, {"hello": "world"})
    assert json.loads(target.read_text(encoding="utf-8"))["hello"] == "world"
