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
from label_master.infra.reporting import build_run_warnings_payload, generate_run_id, persist_run_artifacts
from label_master.reports.schemas import (
    DroppedAnnotationModel,
    RunConfigModel,
    RunReportModel,
    SummaryCountsModel,
)


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
    warnings_path = tmp_path / "reports" / f"{run_id}.warnings.json"
    assert not warnings_path.exists()
    dropped_path = tmp_path / "reports" / f"{run_id}.dropped_annotations.json"
    assert not dropped_path.exists()

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id


def test_reporting_persists_non_empty_warning_and_drop_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_id = generate_run_id()
    config = RunConfigModel(
        run_id=run_id,
        mode="convert",
        input_path="/tmp/in",
        output_path="/tmp/out",
        src_format="coco",
        dst_format="yolo",
        created_at=datetime.now(UTC),
    )
    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path="/tmp/in",
        output_path="/tmp/out",
        summary_counts=SummaryCountsModel(
            images=1,
            annotations_in=2,
            annotations_out=1,
            dropped=1,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
        warnings=[
            {
                "code": "validation_invalid_annotations_dropped",
                "message": "Dropped 1 invalid annotation(s) during permissive validation.",
                "severity": "warning",
                "context": {},
            }
        ],
    )
    dropped_annotations = [
        DroppedAnnotationModel(
            annotation_id="ann-1",
            image_id="img-1",
            image_file="images/a.jpg",
            class_id=0,
            class_name="drone",
            bbox_xywh_abs=(1.0, 2.0, 3.0, 4.0),
            stage="validation",
            reason_code="bbox_out_of_frame",
            reason="bbox goes out of frame",
        )
    ]

    persist_run_artifacts(
        tmp_path / "reports",
        run_id,
        config,
        report,
        dropped_annotations=dropped_annotations,
    )

    warnings_path = tmp_path / "reports" / f"{run_id}.warnings.json"
    dropped_path = tmp_path / "reports" / f"{run_id}.dropped_annotations.json"
    assert warnings_path.exists()
    assert dropped_path.exists()


def test_build_run_warnings_payload_groups_dropped_annotations_by_file() -> None:
    run_id = generate_run_id()
    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path="/tmp/in",
        summary_counts=SummaryCountsModel(
            images=1,
            annotations_in=2,
            annotations_out=1,
            dropped=1,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
    )
    dropped_annotations = [
        DroppedAnnotationModel(
            annotation_id="ann-1",
            image_id="img-1",
            image_file="images/a.jpg",
            class_id=0,
            class_name="drone",
            bbox_xywh_abs=(1.0, 2.0, 3.0, 4.0),
            stage="validation",
            reason_code="bbox_out_of_frame",
            reason="bbox goes out of frame",
        ),
        DroppedAnnotationModel(
            annotation_id="ann-2",
            image_id="img-1",
            image_file="images/a.jpg",
            class_id=0,
            class_name="drone",
            bbox_xywh_abs=(5.0, 6.0, 7.0, 8.0),
            stage="remap",
            reason_code="class_dropped_by_mapping",
            reason="Dropped by class mapping.",
        ),
    ]

    payload = build_run_warnings_payload(report, dropped_annotations=dropped_annotations)

    assert payload["run_id"] == run_id
    assert payload["files_with_infractions_count"] == 1
    assert payload["file_infractions"][0]["file"] == "images/a.jpg"
    assert payload["file_infractions"][0]["infraction_count"] == 2
    assert [item["annotation_id"] for item in payload["file_infractions"][0]["infractions"]] == [
        "ann-1",
        "ann-2",
    ]


def test_build_run_warnings_payload_groups_load_stage_drops_by_source_file() -> None:
    run_id = generate_run_id()
    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path="/tmp/in",
        summary_counts=SummaryCountsModel(
            images=0,
            annotations_in=0,
            annotations_out=0,
            dropped=1,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
    )
    dropped_annotations = [
        DroppedAnnotationModel(
            source_file="val/xml/00991.xml",
            stage="load",
            reason_code="voc_annotation_file_skipped",
            reason="VOC bbox must have xmax > xmin and ymax > ymin: val/xml/00991.xml",
        )
    ]

    payload = build_run_warnings_payload(report, dropped_annotations=dropped_annotations)

    assert payload["files_with_infractions_count"] == 1
    assert payload["file_infractions"][0]["file"] == "val/xml/00991.xml"
    assert payload["file_infractions"][0]["image_id"] is None
    assert payload["file_infractions"][0]["infractions"][0]["stage"] == "load"


def test_atomic_write_json_writes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "nested" / "payload.json"
    atomic_write_json(target, {"hello": "world"})
    assert json.loads(target.read_text(encoding="utf-8"))["hello"] == "world"
