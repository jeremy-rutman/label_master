from __future__ import annotations

from datetime import UTC, datetime

import pytest

from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    InferenceCandidate,
    InferenceResult,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.policies import (
    InferencePolicy,
    InvalidAnnotationAction,
    RemapPolicy,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)


def test_annotation_dataset_rejects_missing_image_reference() -> None:
    with pytest.raises(ValueError):
        AnnotationDataset(
            dataset_id="ds-1",
            source_format=SourceFormat.COCO,
            images=[ImageRecord(image_id="img-1", file_name="x.jpg", width=10, height=10)],
            annotations=[
                AnnotationRecord(
                    annotation_id="ann-1",
                    image_id="missing",
                    class_id=0,
                    bbox_xywh_abs=(1.0, 1.0, 2.0, 2.0),
                )
            ],
            categories={0: CategoryRecord(class_id=0, name="drone")},
            source_metadata=SourceMetadata(
                dataset_root="/tmp/ds",
                loaded_at=datetime.now(UTC),
                loader="unit-test",
            ),
        )


def test_inference_result_requires_descending_scores() -> None:
    with pytest.raises(ValueError):
        InferenceResult(
            predicted_format=SourceFormat.COCO,
            confidence=0.9,
            candidates=[
                InferenceCandidate(format=SourceFormat.COCO, score=0.5),
                InferenceCandidate(format=SourceFormat.YOLO, score=0.9),
            ],
        )


def test_remap_policy_resolution_modes() -> None:
    mapping = {0: 4, 1: None}

    assert RemapPolicy(unmapped_policy=UnmappedPolicy.ERROR).resolve_destination(0, mapping) == 4
    assert RemapPolicy(unmapped_policy=UnmappedPolicy.DROP).resolve_destination(2, mapping) is None
    assert RemapPolicy(unmapped_policy=UnmappedPolicy.IDENTITY).resolve_destination(2, mapping) == 2

    with pytest.raises(ValueError):
        RemapPolicy(unmapped_policy=UnmappedPolicy.ERROR).resolve_destination(2, mapping)


def test_validation_policy_modes() -> None:
    strict = ValidationPolicy.for_mode(ValidationMode.STRICT)
    permissive = ValidationPolicy.for_mode(
        ValidationMode.PERMISSIVE,
        invalid_annotation_action=InvalidAnnotationAction.DROP,
    )

    assert strict.max_invalid_annotations == 0
    assert permissive.max_invalid_annotations > strict.max_invalid_annotations
    assert permissive.invalid_annotation_action == InvalidAnnotationAction.DROP
    assert strict.correct_out_of_frame_bboxes is True
    assert strict.out_of_frame_tolerance_px == 20.0


def test_inference_policy_ambiguity_margin() -> None:
    policy = InferencePolicy(ambiguity_margin=0.05)
    assert policy.is_ambiguous(0.8, 0.76)
    assert not policy.is_ambiguous(0.8, 0.6)
