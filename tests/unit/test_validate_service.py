from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    InvalidAnnotationAction,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.core.services.validate_service import validate_dataset

VIDEO_FIXTURE = Path("tests/fixtures/us3/provider_sample2_video")


def _write_coco_dataset(
    dataset_root: Path,
    *,
    bbox: tuple[float, float, float, float],
) -> None:
    payload = {
        "images": [
            {
                "id": "img-1",
                "file_name": "images/example.jpg",
                "width": 100,
                "height": 50,
            }
        ],
        "annotations": [
            {
                "id": "ann-1",
                "image_id": "img-1",
                "category_id": 0,
                "bbox": list(bbox),
            }
        ],
        "categories": [{"id": 0, "name": "object"}],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def test_validate_dataset_clips_reasonable_bbox_overflow(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(ValidationMode.STRICT),
    )

    assert outcome.summary.valid is True
    assert outcome.summary.invalid_annotations == 0
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)
    assert len(outcome.warnings) == 1
    assert outcome.warnings[0].code == "validation_bbox_clipped_to_frame"
    assert "bbox went slightly out of frame" in outcome.warnings[0].message


def test_validate_dataset_flags_large_bbox_overflow(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(95.0, 40.0, 30.0, 10.0))

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(ValidationMode.PERMISSIVE),
    )

    assert outcome.summary.valid is True
    assert outcome.summary.invalid_annotations == 1
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (95.0, 40.0, 30.0, 10.0)
    assert outcome.warnings == []
    assert "bbox goes out of frame beyond the accepted 20px correction tolerance" in outcome.summary.errors[0]


def test_validate_dataset_can_drop_invalid_annotations_in_permissive_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(95.0, 40.0, 30.0, 10.0))

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(
            ValidationMode.PERMISSIVE,
            invalid_annotation_action=InvalidAnnotationAction.DROP,
        ),
    )

    assert outcome.summary.valid is True
    assert outcome.summary.invalid_annotations == 1
    assert outcome.dataset.annotations == []
    assert len(outcome.warnings) == 1
    assert outcome.warnings[0].code == "validation_invalid_annotations_dropped"
    assert len(outcome.dropped_annotations) == 1
    assert outcome.dropped_annotations[0].annotation_id == "ann-1"
    assert outcome.dropped_annotations[0].stage == "validation"
    assert outcome.dropped_annotations[0].reason_code == "bbox_out_of_frame"
    assert outcome.dropped_annotations[0].image_file == "images/example.jpg"
    assert outcome.dropped_annotations[0].context["overflow_px"] == "right=25.00"


def test_validate_dataset_surfaces_loader_skipped_voc_files_as_dropped_items(tmp_path) -> None:
    xml_dir = tmp_path / "val" / "xml"
    xml_dir.mkdir(parents=True)
    (xml_dir / "00991.xml").write_text(
        """<annotation>
    <size><width>1920</width><height>1080</height></size>
</annotation>
""",
        encoding="utf-8",
    )

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.VOC,
        policy=ValidationPolicy.for_mode(ValidationMode.PERMISSIVE),
    )

    assert outcome.summary.valid is True
    assert outcome.summary.invalid_annotations == 0
    assert len(outcome.dropped_annotations) == 1
    assert outcome.dropped_annotations[0].stage == "load"
    assert outcome.dropped_annotations[0].source_file == "val/xml/00991.xml"
    assert outcome.dropped_annotations[0].reason_code == "voc_annotation_file_skipped"
    assert outcome.dropped_annotations[0].reason == "VOC annotation is missing filename: val/xml/00991.xml"


def test_validate_dataset_strict_error_includes_first_issue(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(95.0, 40.0, 30.0, 10.0))

    with pytest.raises(ValidationError) as exc_info:
        validate_dataset(
            tmp_path,
            source_format=SourceFormat.COCO,
            policy=ValidationPolicy.for_mode(ValidationMode.STRICT),
        )

    assert "Validation failed in strict mode: 1 invalid annotation(s)" in str(exc_info.value)
    assert exc_info.value.context["invalid_annotations"] == "1"
    assert "bbox goes out of frame beyond the accepted 20px correction tolerance" in exc_info.value.context[
        "first_error"
    ]
    assert exc_info.value.context["sample_errors"] == exc_info.value.context["first_error"]
    issue_rows = json.loads(exc_info.value.context["issue_rows_json"])
    assert issue_rows[0]["annotation_id"] == "ann-1"
    assert issue_rows[0]["bbox_xywh_abs"] == "(95.00, 40.00, 30.00, 10.00)"
    assert issue_rows[0]["frame_bounds"] == "width=100, height=50"
    assert issue_rows[0]["overflow_px"] == "right=25.00"


def test_validate_dataset_can_disable_out_of_frame_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(
            ValidationMode.PERMISSIVE,
            correct_out_of_frame_bboxes=False,
        ),
    )

    assert outcome.summary.invalid_annotations == 1
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (91.0, 40.0, 10.0, 10.0)
    assert outcome.warnings == []
    assert "Annotation ann-1 bbox goes out of frame" in outcome.summary.errors[0]
    assert "bbox_xywh_abs=(91.00, 40.00, 10.00, 10.00)" in outcome.summary.errors[0]
    assert "frame_bounds=width=100, height=50" in outcome.summary.errors[0]
    assert "overflow_px=right=1.00" in outcome.summary.errors[0]


def test_validate_dataset_honors_custom_out_of_frame_tolerance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.6, 10.0))

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(
            ValidationMode.STRICT,
            out_of_frame_tolerance_px=2.0,
        ),
    )

    assert outcome.summary.invalid_annotations == 0
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)
    assert len(outcome.warnings) == 1
    assert "<= 2px" in outcome.warnings[0].message


def test_validate_dataset_reports_annotation_progress(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))
    observed: list[tuple[int, int]] = []

    outcome = validate_dataset(
        tmp_path,
        source_format=SourceFormat.COCO,
        policy=ValidationPolicy.for_mode(ValidationMode.STRICT),
        annotation_progress_callback=lambda completed, total: observed.append((completed, total)),
    )

    assert outcome.summary.valid is True
    assert observed == [(1, 1)]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_validate_dataset_clips_partially_out_of_frame_video_bbox_rows(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "video_bbox_negative_y"
    annotations_root = dataset_root / "annotations"
    videos_root = dataset_root / "wosdetc_train_videos"
    annotations_root.mkdir(parents=True)
    videos_root.mkdir(parents=True)

    source_annotation = VIDEO_FIXTURE / "annotations" / "00_02_45_to_00_03_10_cut.txt"
    source_video = VIDEO_FIXTURE / "videos" / "00_02_45_to_00_03_10_cut.mpg"
    shutil.copy2(source_video, videos_root / source_video.name)

    annotation_lines = source_annotation.read_text(encoding="utf-8").splitlines()
    annotation_lines[0] = "0 1 708 -3 15 13 drone"
    (annotations_root / source_annotation.name).write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")

    outcome = validate_dataset(
        dataset_root,
        source_format=SourceFormat.VIDEO_BBOX,
        policy=ValidationPolicy.for_mode(ValidationMode.STRICT),
    )

    assert outcome.summary.invalid_annotations == 0
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (708.0, 0.0, 15.0, 10.0)
    assert len(outcome.warnings) == 1
    assert outcome.warnings[0].code == "validation_bbox_clipped_to_frame"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_validate_dataset_loads_builtin_video_bbox_spec_with_token_mapping() -> None:
    outcome = validate_dataset(
        VIDEO_FIXTURE,
        source_format=SourceFormat.VIDEO_BBOX,
        policy=ValidationPolicy.for_mode(ValidationMode.PERMISSIVE),
    )

    assert outcome.summary.invalid_annotations == 0
    assert outcome.dataset.images[0].file_name == "images/00_02_45_to_00_03_10_cut/frame_000000.jpg"
    assert outcome.dataset.annotations[0].bbox_xywh_abs == (708.0, 757.0, 15.0, 13.0)
