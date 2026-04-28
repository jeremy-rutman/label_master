from __future__ import annotations

import json
from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    InvalidAnnotationAction,
    UnmappedPolicy,
    ValidationMode,
)
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.infra.config import load_mapping_file


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


def test_conversion_success_rate_at_least_95_percent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scenarios = [
        ConvertRequest(
            run_id="scn-1",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn1",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-2",
            input_path=Path("tests/fixtures/us1/yolo_minimal"),
            output_path=tmp_path / "scn2",
            src_format=SourceFormat.YOLO,
            dst_format=SourceFormat.COCO,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-3",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn3",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            class_map=load_mapping_file(Path("tests/fixtures/us1/maps/class_map_drop.yaml")),
            unmapped_policy=UnmappedPolicy.DROP,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-4",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn4",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=True,
        ),
    ]

    successes = 0
    for scenario in scenarios:
        try:
            result = execute_conversion(scenario)
            if result.report.status == "completed":
                successes += 1
        except Exception:
            pass

    rate = successes / len(scenarios)
    assert rate >= 0.95, f"conversion success rate below target: {rate:.3%}"


def test_dry_run_honors_permissive_validation_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "coco_invalid"
    _write_coco_dataset(source, bbox=(95.0, 40.0, 30.0, 10.0))

    result = execute_conversion(
        ConvertRequest(
            run_id="dry-run-permissive-invalid",
            input_path=source,
            output_path=tmp_path / "out",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=True,
            validation_mode=ValidationMode.PERMISSIVE,
        )
    )

    assert result.report.status == "completed"
    assert result.report.summary_counts.invalid == 1
    assert len(result.output_dataset.annotations) == 1


def test_dry_run_can_drop_invalid_annotations_in_permissive_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "coco_invalid_drop"
    _write_coco_dataset(source, bbox=(95.0, 40.0, 30.0, 10.0))

    result = execute_conversion(
        ConvertRequest(
            run_id="dry-run-permissive-invalid-drop",
            input_path=source,
            output_path=tmp_path / "out_drop",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=True,
            validation_mode=ValidationMode.PERMISSIVE,
            permissive_invalid_annotation_action=InvalidAnnotationAction.DROP,
        )
    )

    assert result.report.status == "completed"
    assert result.report.summary_counts.invalid == 1
    assert result.output_dataset.annotations == []
