from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import label_master.interfaces.gui.viewmodels as gui_viewmodels
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.missing_label_hint_service import (
    CombinedYoloDetectorReviewResult,
    HintDetection,
    MissingLabelHint,
    MissingLabelHintResult,
    YoloBBoxAuditItem,
    YoloBBoxAuditProposal,
    YoloBBoxAuditResult,
)
from label_master.interfaces.gui.viewmodels import (
    BBoxAuditItemViewModel,
    BBoxAuditProposalViewModel,
    DetectorReviewEditorBoxViewModel,
    DetectorReviewItemViewModel,
    MappingRowViewModel,
    MissingLabelHintDetectionViewModel,
    MissingLabelHintItemViewModel,
    _preview_dataset_view_cached,
    apply_detector_review_bbox_strategy,
    approve_bbox_audit_view,
    approve_detector_review_edited_boxes_view,
    approve_detector_review_view,
    approve_missing_label_hints_view,
    build_detector_review_editor_state_view,
    build_detector_review_overlay_labels,
    build_gui_run_config,
    build_missing_label_hint_overlay_labels,
    convert_view,
    detector_review_final_bbox_xywh_normalized,
    generate_bbox_audit_view,
    generate_detector_review_item_view,
    generate_detector_review_view,
    generate_missing_label_hints_view,
    infer_view,
    list_detector_review_image_paths_view,
    parse_mapping_rows,
    preview_dataset_view,
)


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


def _write_yolo_dataset(
    dataset_root: Path,
    *,
    rows: list[str],
    image_size: tuple[int, int] = (100, 50),
    image_rel: str = "train/images/example.jpg",
    label_rel: str = "train/labels/example.txt",
) -> None:
    image_path = dataset_root / image_rel
    label_path = dataset_root / label_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color="black").save(image_path)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("object\n", encoding="utf-8")


def _write_anti_uav_preview_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "Anti_UAV_RGBT"
    for clip_name, rows in {
        "clip_a": {"exist": [1, 0], "gt_rect": [[10, 20, 30, 40], []]},
        "clip_b": {"exist": [0, 1], "gt_rect": [[], [5, 6, 7, 8]]},
    }.items():
        clip_root = dataset_root / "train" / clip_name
        clip_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("tests/fixtures/us9/V_BIRD_029.mp4"), clip_root / "visible.mp4")
        (clip_root / "visible.json").write_text(json.dumps(rows), encoding="utf-8")
    return dataset_root


def test_parse_mapping_rows_valid_entries() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="2", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="map", destination_class_id="10"),
            MappingRowViewModel(source_class_id="4", action="map", destination_class_id="11"),
        ]
    )

    assert parsed.errors == []
    assert parsed.class_map == {2: None, 3: 10, 4: 11}


def test_parse_mapping_rows_validation_errors() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="foo", action="map", destination_class_id="1"),
            MappingRowViewModel(source_class_id="1", action="map", destination_class_id=""),
            MappingRowViewModel(source_class_id="1", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="2", action="invalid", destination_class_id=""),
        ]
    )

    assert parsed.errors == [
        "Row 1: source_class_id must be an integer",
        "Row 2: destination_class_id is required when action is 'map'",
        "Row 3: duplicate source_class_id 1",
        "Row 4: action must be 'map', 'drop', or 'drop_frame'",
    ]
    assert parsed.class_map == {}


def test_parse_mapping_rows_invalid_destination_integer() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="9", action="map", destination_class_id="cat"),
        ]
    )

    assert parsed.errors == ["Row 1: destination_class_id must be an integer"]
    assert parsed.class_map == {}


def test_parse_mapping_rows_ignores_blank_trailing_rows() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="", action="map", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="", action="map", destination_class_id=""),
        ]
    )

    assert parsed.errors == []
    assert parsed.class_map == {3: None}


def test_preview_dataset_view_coco_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us1/coco_minimal"),
        source_format="coco",
    )

    assert preview.source_format == "coco"
    assert preview.image_count == len(preview.images)
    assert preview.image_count > 0
    assert preview.images[0].bboxes


def test_preview_dataset_view_yolo_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us1/yolo_minimal"),
        source_format="yolo",
    )

    assert preview.source_format == "yolo"
    assert preview.image_count == len(preview.images)
    assert preview.image_count > 0
    assert preview.images[0].bboxes


def test_preview_dataset_view_yolo_sidecar_img_labels_with_obj_names(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 0.2 0.4"],
        image_rel="train/img/example.jpg",
        label_rel="train/img/example.txt",
    )
    (tmp_path / "classes.txt").unlink()
    (tmp_path / "train").mkdir(exist_ok=True)
    (tmp_path / "train" / "obj.names").write_text("drone\n", encoding="utf-8")

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.source_format == "yolo"
    assert preview.images[0].file_name == "train/img/example.jpg"
    assert preview.images[0].bboxes[0].class_name == "drone"


def test_preview_dataset_view_yolo_clips_slightly_out_of_range_normalized_bbox(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 1.01 0.5 0.2 0.4"],
    )

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 20px)."
    ]
    assert preview.images[0].file_name == "train/images/example.jpg"
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 15.0, 9.0, 20.0)


def test_preview_dataset_view_yolo_reports_large_out_of_range_normalized_bbox(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 1.6 0.4"],
    )

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.warnings == [
        "Preview loaded with 1 invalid annotation(s): bbox goes out of frame beyond the accepted 20px correction tolerance."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (-30.0, 15.0, 160.0, 20.0)


def test_generate_missing_label_hints_view_returns_detailed_hints(
    monkeypatch,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    result = MissingLabelHintResult(
        model_path=tmp_path / "model.pt",
        hints_output_dir=tmp_path / "hints",
        report_path=tmp_path / "hints" / "missing_label_hints.report.json",
        scanned_images=4,
        images_with_existing_labels=2,
        missing_label_images=2,
        hinted_images=1,
        hint_files_written=1,
        total_detections=2,
        hints=[
            MissingLabelHint(
                image_rel_path="train/images/example.jpg",
                suggested_label_rel_path="train/labels/example.txt",
                detections=[
                    HintDetection(
                        class_id=0,
                        class_name="drone",
                        confidence=0.92,
                        bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                    ),
                    HintDetection(
                        class_id=1,
                        class_name="bird",
                        confidence=0.51,
                        bbox_xywh_normalized=(0.25, 0.4, 0.1, 0.2),
                    ),
                ],
            )
        ],
    )

    monkeypatch.setattr(gui_viewmodels, "generate_missing_yolo_label_hints", lambda **kwargs: result)

    vm = generate_missing_label_hints_view(
        input_path=tmp_path,
        source_format="yolo",
        detector_model_path=tmp_path / "model.pt",
        hints_output_dir=tmp_path / "hints",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections_per_image=200,
    )

    assert vm.hinted_images == 1
    assert vm.hint_files_written == 1
    assert vm.sample_hints == [
        {
            "image": "train/images/example.jpg",
            "suggested_label": "train/labels/example.txt",
            "detections": 2,
        }
    ]
    assert len(vm.hints) == 1
    assert vm.hints[0].suggested_label_rel_path == "train/labels/example.txt"
    assert [item.class_name for item in vm.hints[0].detections] == ["drone", "bird"]


def test_generate_missing_label_hints_view_rejects_non_yolo_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="YOLO source datasets only"):
        generate_missing_label_hints_view(
            input_path=tmp_path,
            source_format="coco",
            detector_model_path=tmp_path / "model.pt",
            hints_output_dir=tmp_path / "hints",
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections_per_image=200,
        )


def test_build_missing_label_hint_overlay_labels_converts_to_absolute_boxes(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    image_path = tmp_path / "train" / "images" / "example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(image_path)

    overlay_labels = build_missing_label_hint_overlay_labels(
        dataset_root=tmp_path,
        hint=MissingLabelHintItemViewModel(
            image_rel_path="train/images/example.jpg",
            suggested_label_rel_path="train/labels/example.txt",
            detections=[
                MissingLabelHintDetectionViewModel(
                    class_id=0,
                    class_name="drone",
                    confidence=0.92,
                    bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                )
            ],
        ),
    )

    assert overlay_labels == [(40.0, 15.0, 20.0, 20.0, "0:drone 0.92")]


def test_build_detector_review_overlay_labels_keeps_existing_annotations_without_proposals(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    image_path = tmp_path / "train" / "images" / "example.jpg"
    label_path = tmp_path / "train" / "labels" / "example.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(image_path)
    label_path.write_text("0 0.500000 0.500000 0.200000 0.400000\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("drone\n", encoding="utf-8")

    overlay_labels = build_detector_review_overlay_labels(
        dataset_root=tmp_path,
        review_item=DetectorReviewItemViewModel(
            source_kind="bbox_audit",
            image_rel_path="train/images/example.jpg",
            label_rel_path="train/labels/example.txt",
            existing_label_count=1,
            proposals=[],
        ),
        selected_proposals=[],
    )

    assert overlay_labels == [
        (40.0, 15.0, 20.0, 20.0, "ann 0:drone", "blue"),
    ]


def test_build_detector_review_overlay_labels_separates_annotation_and_detector_boxes(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    image_path = tmp_path / "train" / "images" / "example.jpg"
    label_path = tmp_path / "train" / "labels" / "example.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(image_path)
    label_path.write_text("0 0.500000 0.500000 0.200000 0.400000\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("drone\n", encoding="utf-8")

    proposal = BBoxAuditProposalViewModel(
        proposal_id="adjust:0:0",
        action="adjust",
        class_id=0,
        class_name="drone",
        confidence=0.91,
        match_iou=0.65,
        existing_label_index=0,
        existing_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
        proposed_bbox_xywh_normalized=(0.55, 0.55, 0.2, 0.4),
    )

    overlay_labels = build_detector_review_overlay_labels(
        dataset_root=tmp_path,
        review_item=DetectorReviewItemViewModel(
            source_kind="bbox_audit",
            image_rel_path="train/images/example.jpg",
            label_rel_path="train/labels/example.txt",
            existing_label_count=1,
            proposals=[proposal],
        ),
        selected_proposals=[proposal],
    )

    assert overlay_labels == [
        (40.0, 15.0, 20.0, 20.0, "ann 0:drone", "blue"),
        (45.00000000000001, 17.500000000000004, 20.0, 20.0, "det adjust 0:drone 0.91", "orange"),
    ]


def test_apply_detector_review_bbox_strategy_smallest_drops_noop_adjust() -> None:
    proposals = [
        BBoxAuditProposalViewModel(
            proposal_id="adjust:0:0",
            action="adjust",
            class_id=0,
            class_name="drone",
            confidence=0.91,
            match_iou=0.65,
            existing_label_index=0,
            existing_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.2),
            proposed_bbox_xywh_normalized=(0.5, 0.5, 0.4, 0.4),
        )
    ]

    assert apply_detector_review_bbox_strategy(proposals, strategy="smallest") == []


def test_apply_detector_review_bbox_strategy_closest_bounds_intersects_boxes() -> None:
    proposals = [
        BBoxAuditProposalViewModel(
            proposal_id="adjust:0:0",
            action="adjust",
            class_id=0,
            class_name="drone",
            confidence=0.91,
            match_iou=0.65,
            existing_label_index=0,
            existing_bbox_xywh_normalized=(0.5, 0.5, 0.4, 0.4),
            proposed_bbox_xywh_normalized=(0.55, 0.5, 0.4, 0.4),
        )
    ]

    transformed = apply_detector_review_bbox_strategy(proposals, strategy="closest_bounds")

    assert len(transformed) == 1
    assert transformed[0].proposed_bbox_xywh_normalized == (0.525, 0.5, 0.3499999999999999, 0.39999999999999997)


def test_detector_review_final_bbox_smallest_prefers_smaller_annotation_box() -> None:
    proposal = BBoxAuditProposalViewModel(
        proposal_id="adjust:0:0",
        action="adjust",
        class_id=0,
        class_name="drone",
        confidence=0.91,
        match_iou=0.65,
        existing_label_index=0,
        existing_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.2),
        proposed_bbox_xywh_normalized=(0.5, 0.5, 0.4, 0.4),
    )

    assert detector_review_final_bbox_xywh_normalized(proposal, strategy="smallest") == (
        0.5,
        0.5,
        0.2,
        0.2,
    )


def test_detector_review_final_bbox_remove_returns_none() -> None:
    proposal = BBoxAuditProposalViewModel(
        proposal_id="remove:0",
        action="remove",
        class_id=0,
        class_name="drone",
        confidence=None,
        match_iou=None,
        existing_label_index=0,
        existing_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.2),
        proposed_bbox_xywh_normalized=None,
    )

    assert detector_review_final_bbox_xywh_normalized(proposal, strategy="detector") is None


def test_build_detector_review_editor_state_view_applies_selected_proposals(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.500000 0.500000 0.200000 0.400000"],
    )
    (tmp_path / "classes.txt").write_text("drone\nbird\n", encoding="utf-8")

    adjust_proposal = BBoxAuditProposalViewModel(
        proposal_id="adjust:0:0",
        action="adjust",
        class_id=0,
        class_name="drone",
        confidence=0.91,
        match_iou=0.65,
        existing_label_index=0,
        existing_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
        proposed_bbox_xywh_normalized=(0.55, 0.55, 0.2, 0.4),
    )
    add_proposal = BBoxAuditProposalViewModel(
        proposal_id="add:1",
        action="add",
        class_id=1,
        class_name="bird",
        confidence=0.88,
        match_iou=None,
        existing_label_index=None,
        existing_bbox_xywh_normalized=None,
        proposed_bbox_xywh_normalized=(0.2, 0.2, 0.1, 0.1),
    )

    editor_state = build_detector_review_editor_state_view(
        dataset_root=tmp_path,
        review_item=DetectorReviewItemViewModel(
            source_kind="bbox_audit",
            image_rel_path="train/images/example.jpg",
            label_rel_path="train/labels/example.txt",
            existing_label_count=1,
            proposals=[adjust_proposal, add_proposal],
        ),
        selected_proposals=[adjust_proposal, add_proposal],
        strategy="detector",
    )

    assert [(box.class_id, box.class_name) for box in editor_state.annotation_boxes] == [(0, "drone")]
    assert [(box.class_id, box.class_name) for box in editor_state.detector_boxes] == [
        (0, "drone"),
        (1, "bird"),
    ]
    assert [(box.class_id, box.class_name, box.bbox_xywh_normalized) for box in editor_state.editable_boxes] == [
        (0, "drone", (0.55, 0.55, 0.2, 0.4)),
        (1, "bird", (0.2, 0.2, 0.1, 0.1)),
    ]
    assert [(option.class_id, option.class_name) for option in editor_state.class_options] == [
        (0, "drone"),
        (1, "bird"),
    ]


def test_approve_detector_review_edited_boxes_view_writes_final_boxes(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    image_path = tmp_path / "train" / "images" / "example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(image_path)
    (tmp_path / "classes.txt").write_text("drone\n", encoding="utf-8")

    approval = approve_detector_review_edited_boxes_view(
        dataset_root=tmp_path,
        review_item=DetectorReviewItemViewModel(
            source_kind="missing_label",
            image_rel_path="train/images/example.jpg",
            label_rel_path="train/labels/example.txt",
            existing_label_count=0,
            proposals=[],
        ),
        edited_boxes=[
            DetectorReviewEditorBoxViewModel(
                box_id="manual:1",
                class_id=0,
                class_name="drone",
                bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                source="manual",
            )
        ],
        allow_overwrite_missing_label_files=False,
    )

    label_path = tmp_path / "train" / "labels" / "example.txt"
    assert approval.approved_label_files == 1
    assert approval.applied_proposals == 1
    assert approval.label_paths == [str(label_path)]
    assert label_path.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.200000 0.400000\n"


def test_approve_missing_label_hints_view_writes_selected_detections(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    approval = approve_missing_label_hints_view(
        dataset_root=tmp_path,
        approved_hints=[
            MissingLabelHintItemViewModel(
                image_rel_path="train/images/example.jpg",
                suggested_label_rel_path="train/labels/example.txt",
                detections=[
                    MissingLabelHintDetectionViewModel(
                        class_id=0,
                        class_name="drone",
                        confidence=0.92,
                        bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                    ),
                    MissingLabelHintDetectionViewModel(
                        class_id=1,
                        class_name="bird",
                        confidence=0.51,
                        bbox_xywh_normalized=(0.25, 0.4, 0.1, 0.2),
                    ),
                ],
            )
        ],
    )

    label_path = tmp_path / "train" / "labels" / "example.txt"

    assert approval.approved_label_files == 1
    assert approval.approved_detections == 2
    assert approval.label_paths == [str(label_path)]
    assert label_path.read_text(encoding="utf-8") == (
        "0 0.500000 0.500000 0.200000 0.400000\n"
        "1 0.250000 0.400000 0.100000 0.200000\n"
    )


def test_approve_missing_label_hints_view_rejects_overwrite_without_opt_in(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    label_path = tmp_path / "train" / "labels" / "example.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing label file"):
        approve_missing_label_hints_view(
            dataset_root=tmp_path,
            approved_hints=[
                MissingLabelHintItemViewModel(
                    image_rel_path="train/images/example.jpg",
                    suggested_label_rel_path="train/labels/example.txt",
                    detections=[
                        MissingLabelHintDetectionViewModel(
                            class_id=0,
                            class_name="drone",
                            confidence=0.92,
                            bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                        )
                    ],
                )
            ],
        )


def test_generate_bbox_audit_view_returns_detailed_items(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = SimpleNamespace(
        report_output_dir=tmp_path / "audit",
        report_path=tmp_path / "audit" / "yolo_bbox_audit.report.json",
        scanned_labeled_images=4,
        images_with_proposals=1,
        add_proposals=1,
        remove_proposals=1,
        adjust_proposals=1,
        items=[
            SimpleNamespace(
                image_rel_path="train/images/example.jpg",
                label_rel_path="train/labels/example.txt",
                existing_label_count=2,
                proposals=[
                    SimpleNamespace(
                        proposal_id="remove:1",
                        action="remove",
                        class_id=1,
                        class_name="bird",
                        confidence=None,
                        match_iou=None,
                        existing_label_index=1,
                        existing_bbox_xywh_normalized=(0.2, 0.2, 0.1, 0.1),
                        proposed_bbox_xywh_normalized=None,
                    ),
                    SimpleNamespace(
                        proposal_id="add:0",
                        action="add",
                        class_id=2,
                        class_name="plane",
                        confidence=0.88,
                        match_iou=None,
                        existing_label_index=None,
                        existing_bbox_xywh_normalized=None,
                        proposed_bbox_xywh_normalized=(0.8, 0.8, 0.1, 0.1),
                    ),
                ],
            )
        ],
    )

    monkeypatch.setattr(gui_viewmodels, "generate_yolo_bbox_audit", lambda **kwargs: result)

    vm = generate_bbox_audit_view(
        input_path=tmp_path,
        source_format="yolo",
        detector_model_path=tmp_path / "model.pt",
        report_output_dir=tmp_path / "audit",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections_per_image=200,
        max_labeled_images=100,
        match_iou_threshold=0.30,
        correction_iou_threshold=0.85,
    )

    assert vm.images_with_proposals == 1
    assert vm.add_proposals == 1
    assert vm.remove_proposals == 1
    assert vm.adjust_proposals == 1
    assert len(vm.items) == 1
    assert vm.items[0].existing_label_count == 2
    assert [proposal.action for proposal in vm.items[0].proposals] == ["remove", "add"]


def test_generate_detector_review_view_returns_combined_results(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = CombinedYoloDetectorReviewResult(
        missing_label_hints=MissingLabelHintResult(
            model_path=tmp_path / "model.pt",
            hints_output_dir=tmp_path / "hints",
            report_path=tmp_path / "hints" / "missing_label_hints.report.json",
            scanned_images=4,
            images_with_existing_labels=2,
            missing_label_images=2,
            hinted_images=1,
            hint_files_written=1,
            total_detections=2,
            hints=[
                MissingLabelHint(
                    image_rel_path="train/images/unlabeled.jpg",
                    suggested_label_rel_path="train/labels/unlabeled.txt",
                    detections=[
                        HintDetection(
                            class_id=0,
                            class_name="drone",
                            confidence=0.92,
                            bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                        )
                    ],
                )
            ],
        ),
        bbox_audit=YoloBBoxAuditResult(
            model_path=tmp_path / "model.pt",
            report_output_dir=tmp_path / "audit",
            report_path=tmp_path / "audit" / "yolo_bbox_audit.report.json",
            scanned_labeled_images=4,
            images_with_proposals=1,
            add_proposals=1,
            remove_proposals=1,
            adjust_proposals=0,
            items=[
                YoloBBoxAuditItem(
                    image_rel_path="train/images/example.jpg",
                    label_rel_path="train/labels/example.txt",
                    existing_label_count=2,
                    proposals=[
                        YoloBBoxAuditProposal(
                            proposal_id="add:0",
                            action="add",
                            class_id=2,
                            class_name="plane",
                            confidence=0.88,
                            match_iou=None,
                            existing_label_index=None,
                            existing_bbox_xywh_normalized=None,
                            proposed_bbox_xywh_normalized=(0.8, 0.8, 0.1, 0.1),
                        )
                    ],
                )
            ],
        ),
    )

    monkeypatch.setattr(gui_viewmodels, "generate_yolo_detector_review", lambda **kwargs: result)

    vm = generate_detector_review_view(
        input_path=tmp_path,
        source_format="yolo",
        detector_model_path=tmp_path / "model.pt",
        hints_output_dir=tmp_path / "hints",
        report_output_dir=tmp_path / "audit",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections_per_image=200,
        max_labeled_images=100,
        match_iou_threshold=0.30,
        correction_iou_threshold=0.85,
    )

    assert vm.missing_label_hints.hinted_images == 1
    assert vm.missing_label_hints.hints[0].suggested_label_rel_path == "train/labels/unlabeled.txt"
    assert vm.bbox_audit.images_with_proposals == 1
    assert vm.bbox_audit.items[0].proposals[0].action == "add"
    assert len(vm.items) == 2
    assert {item.source_kind for item in vm.items} == {"missing_label", "bbox_audit"}
    missing_item = next(item for item in vm.items if item.source_kind == "missing_label")
    bbox_item = next(item for item in vm.items if item.source_kind == "bbox_audit")
    assert missing_item.label_rel_path == "train/labels/unlabeled.txt"
    assert missing_item.proposals[0].action == "add"
    assert bbox_item.label_rel_path == "train/labels/example.txt"


def test_list_detector_review_image_paths_view_delegates_for_yolo(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        gui_viewmodels,
        "discover_yolo_review_image_paths",
        lambda **kwargs: ["train/images/example.jpg", "train/images/unlabeled.jpg"],
    )

    image_paths = list_detector_review_image_paths_view(
        input_path=tmp_path,
        source_format="yolo",
        input_path_include_substring="train",
        input_path_exclude_substring="backup",
    )

    assert image_paths == ["train/images/example.jpg", "train/images/unlabeled.jpg"]


def test_generate_detector_review_item_view_maps_single_image_review(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        gui_viewmodels,
        "generate_yolo_detector_review_for_image",
        lambda **kwargs: SimpleNamespace(
            image_rel_path="train/images/example.jpg",
            label_rel_path="train/labels/example.txt",
            has_existing_label_file=True,
            existing_label_count=2,
            proposals=[
                SimpleNamespace(
                    proposal_id="add:0",
                    action="add",
                    class_id=2,
                    class_name="plane",
                    confidence=0.88,
                    match_iou=None,
                    existing_label_index=None,
                    existing_bbox_xywh_normalized=None,
                    proposed_bbox_xywh_normalized=(0.8, 0.8, 0.1, 0.1),
                )
            ],
        ),
    )

    item = generate_detector_review_item_view(
        input_path=tmp_path,
        source_format="yolo",
        image_rel_path="train/images/example.jpg",
        detector_model_path=tmp_path / "model.pt",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections_per_image=200,
        match_iou_threshold=0.30,
        correction_iou_threshold=0.85,
    )

    assert item.source_kind == "bbox_audit"
    assert item.label_rel_path == "train/labels/example.txt"
    assert item.existing_label_count == 2
    assert [proposal.action for proposal in item.proposals] == ["add"]


def test_approve_detector_review_view_splits_missing_and_audit_items(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}

    def _fake_approve_missing_label_hints_view(*, dataset_root, approved_hints, allow_overwrite):  # type: ignore[no-untyped-def]
        observed["missing_dataset_root"] = dataset_root
        observed["approved_hints"] = approved_hints
        observed["allow_overwrite"] = allow_overwrite
        return SimpleNamespace(
            approved_label_files=1,
            approved_detections=1,
            label_paths=[str(tmp_path / "train" / "labels" / "unlabeled.txt")],
        )

    def _fake_approve_bbox_audit_view(*, dataset_root, approved_items):  # type: ignore[no-untyped-def]
        observed["bbox_dataset_root"] = dataset_root
        observed["approved_items"] = approved_items
        return SimpleNamespace(
            approved_label_files=1,
            applied_proposals=2,
            label_paths=[str(tmp_path / "train" / "labels" / "example.txt")],
        )

    monkeypatch.setattr(
        gui_viewmodels,
        "approve_missing_label_hints_view",
        _fake_approve_missing_label_hints_view,
    )
    monkeypatch.setattr(gui_viewmodels, "approve_bbox_audit_view", _fake_approve_bbox_audit_view)

    approval = approve_detector_review_view(
        dataset_root=tmp_path,
        approved_items=[
            DetectorReviewItemViewModel(
                source_kind="missing_label",
                image_rel_path="train/images/unlabeled.jpg",
                label_rel_path="train/labels/unlabeled.txt",
                existing_label_count=0,
                proposals=[
                    BBoxAuditProposalViewModel(
                        proposal_id="hint:add:0",
                        action="add",
                        class_id=0,
                        class_name="drone",
                        confidence=0.92,
                        match_iou=None,
                        existing_label_index=None,
                        existing_bbox_xywh_normalized=None,
                        proposed_bbox_xywh_normalized=(0.5, 0.5, 0.2, 0.4),
                    )
                ],
            ),
            DetectorReviewItemViewModel(
                source_kind="bbox_audit",
                image_rel_path="train/images/example.jpg",
                label_rel_path="train/labels/example.txt",
                existing_label_count=2,
                proposals=[
                    BBoxAuditProposalViewModel(
                        proposal_id="remove:1",
                        action="remove",
                        class_id=1,
                        class_name="bird",
                        confidence=None,
                        match_iou=None,
                        existing_label_index=1,
                        existing_bbox_xywh_normalized=(0.2, 0.2, 0.1, 0.1),
                        proposed_bbox_xywh_normalized=None,
                    ),
                    BBoxAuditProposalViewModel(
                        proposal_id="add:0",
                        action="add",
                        class_id=2,
                        class_name="plane",
                        confidence=0.88,
                        match_iou=None,
                        existing_label_index=None,
                        existing_bbox_xywh_normalized=None,
                        proposed_bbox_xywh_normalized=(0.8, 0.8, 0.1, 0.1),
                    ),
                ],
            ),
        ],
        allow_overwrite_missing_label_files=True,
    )

    assert observed["missing_dataset_root"] == tmp_path
    assert observed["bbox_dataset_root"] == tmp_path
    assert observed["allow_overwrite"] is True
    assert len(observed["approved_hints"]) == 1
    assert observed["approved_hints"][0].suggested_label_rel_path == "train/labels/unlabeled.txt"
    assert len(observed["approved_items"]) == 1
    assert approval.approved_label_files == 2
    assert approval.applied_proposals == 3


def test_approve_bbox_audit_view_delegates_to_service(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}

    def _fake_apply_yolo_bbox_audit(*, dataset_root, approved_items):  # type: ignore[no-untyped-def]
        observed["dataset_root"] = dataset_root
        observed["approved_items"] = approved_items
        return SimpleNamespace(
            approved_label_files=1,
            applied_proposals=2,
            label_paths=[str(tmp_path / "train" / "labels" / "example.txt")],
        )

    monkeypatch.setattr(gui_viewmodels, "apply_yolo_bbox_audit", _fake_apply_yolo_bbox_audit)

    approval = approve_bbox_audit_view(
        dataset_root=tmp_path,
        approved_items=[
            BBoxAuditItemViewModel(
                image_rel_path="train/images/example.jpg",
                label_rel_path="train/labels/example.txt",
                existing_label_count=2,
                proposals=[
                    BBoxAuditProposalViewModel(
                        proposal_id="remove:1",
                        action="remove",
                        class_id=1,
                        class_name="bird",
                        confidence=None,
                        match_iou=None,
                        existing_label_index=1,
                        existing_bbox_xywh_normalized=(0.2, 0.2, 0.1, 0.1),
                        proposed_bbox_xywh_normalized=None,
                    ),
                    BBoxAuditProposalViewModel(
                        proposal_id="add:0",
                        action="add",
                        class_id=2,
                        class_name="plane",
                        confidence=0.88,
                        match_iou=None,
                        existing_label_index=None,
                        existing_bbox_xywh_normalized=None,
                        proposed_bbox_xywh_normalized=(0.8, 0.8, 0.1, 0.1),
                    ),
                ],
            )
        ],
    )

    assert observed["dataset_root"] == tmp_path
    assert len(observed["approved_items"]) == 1
    assert approval.approved_label_files == 1
    assert approval.applied_proposals == 2


def test_preview_dataset_view_kitware_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us4"),
        source_format="kitware",
    )

    assert preview.source_format == "kitware"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 3673
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "airplane"


def test_preview_dataset_view_voc_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us6"),
        source_format="voc",
    )

    assert preview.source_format == "voc"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 5
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "UAV"
    assert preview.warnings == []


def test_preview_dataset_view_voc_caps_large_preview_sample(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _preview_dataset_view_cached.cache_clear()
    monkeypatch.setattr(gui_viewmodels, "VOC_PREVIEW_MAX_XML_FILES", 2)
    monkeypatch.setattr(gui_viewmodels, "VOC_PREVIEW_WARNING", "preview capped for test")

    preview = preview_dataset_view(
        Path("tests/fixtures/us6"),
        source_format="voc",
    )

    assert preview.source_format == "voc"
    assert preview.image_count == 2
    assert preview.warnings == ["preview capped for test"]

    _preview_dataset_view_cached.cache_clear()


def test_preview_dataset_view_video_bbox_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us5"),
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 183
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "object"


def test_preview_dataset_view_mot_video_bbox_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us7"),
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 317
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "object"
    assert preview.warnings == []


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_preview_dataset_view_video_bbox_caps_large_source_sets(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _preview_dataset_view_cached.cache_clear()
    monkeypatch.setattr(gui_viewmodels, "VIDEO_BBOX_PREVIEW_MAX_SOURCES", 1)
    monkeypatch.setattr(gui_viewmodels, "VIDEO_BBOX_PREVIEW_WARNING", "video preview capped for test")

    dataset_root = _write_anti_uav_preview_dataset(tmp_path)
    preview = preview_dataset_view(
        dataset_root,
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == 2
    assert preview.warnings == ["video preview capped for test"]

    _preview_dataset_view_cached.cache_clear()


def test_preview_dataset_view_warns_when_bbox_is_auto_clipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    preview = preview_dataset_view(tmp_path, source_format="coco")

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 20px)."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)


def test_preview_dataset_view_explains_out_of_frame_invalid_annotations(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(95.0, 40.0, 30.0, 10.0))

    preview = preview_dataset_view(tmp_path, source_format="coco")

    assert preview.warnings == [
        "Preview loaded with 1 invalid annotation(s): bbox goes out of frame beyond the accepted 20px correction tolerance."
    ]


def test_preview_dataset_view_can_disable_out_of_frame_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    preview = preview_dataset_view(
        tmp_path,
        source_format="coco",
        out_of_frame_bbox_policy="ignore",
    )

    # The ignore policy keeps out-of-frame boxes untouched and does not flag them.
    assert preview.warnings == []
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 10.0, 10.0)


def test_preview_dataset_view_honors_custom_out_of_frame_tolerance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.6, 10.0))

    preview = preview_dataset_view(
        tmp_path,
        source_format="coco",
        out_of_frame_tolerance_px=2.0,
    )

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 2px)."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)


def test_preview_dataset_view_yolo_with_incomplete_classes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_root = tmp_path / "yolo_missing_classes"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "0.txt").write_text("4 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("class_zero_only\n", encoding="utf-8")

    preview = preview_dataset_view(dataset_root, source_format="yolo")

    assert preview.source_format == "yolo"
    assert preview.images
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_id == 4
    assert preview.images[0].bboxes[0].class_name == "class_4"


def test_preview_dataset_view_yolo_uses_actual_image_size_when_missing_manifest(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    dataset_root = tmp_path / "yolo_missing_sizes"
    labels_dir = dataset_root / "labels"
    images_dir = dataset_root / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    Image.new("RGB", (100, 80), color="black").save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("class_zero\n", encoding="utf-8")

    preview = preview_dataset_view(dataset_root, source_format="yolo")
    bbox = preview.images[0].bboxes[0]

    assert bbox.bbox_xywh_abs[2] == 20.0
    assert bbox.bbox_xywh_abs[3] == 20.0


def test_preview_dataset_view_reuses_cached_result(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    fake_validation = SimpleNamespace(
        dataset=SimpleNamespace(
            annotations=[],
            categories={},
            images=[
                SimpleNamespace(
                    image_id="image-1",
                    file_name="images/example.jpg",
                    width=100,
                    height=80,
                )
            ],
        ),
        summary=SimpleNamespace(invalid_annotations=0),
    )

    _preview_dataset_view_cached.cache_clear()

    def _fake_validate_dataset(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        calls.append("validate")
        return fake_validation

    monkeypatch.setattr(gui_viewmodels, "validate_dataset", _fake_validate_dataset)

    preview_one = preview_dataset_view(tmp_path, source_format="coco")
    preview_two = preview_dataset_view(tmp_path, source_format="coco")

    assert preview_one == preview_two
    assert calls == ["validate"]

    _preview_dataset_view_cached.cache_clear()


def test_infer_view_uses_fast_gui_sample_limit_by_default(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, int] = {}

    def _fake_infer_format(input_path, *, policy, force):  # type: ignore[no-untyped-def]
        del input_path, force
        observed["sample_limit"] = policy.sample_limit
        return SimpleNamespace(
            predicted_format=SourceFormat.KITWARE,
            confidence=1.0,
            candidates=[SimpleNamespace(format=SourceFormat.KITWARE, score=1.0)],
            warnings=[],
        )

    monkeypatch.setattr(gui_viewmodels, "infer_format", _fake_infer_format)

    vm = infer_view(tmp_path)

    assert observed["sample_limit"] == 100
    assert vm.predicted_format == "kitware"


def test_infer_view_detects_yolo_sidecar_img_layout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 0.2 0.4"],
        image_rel="train/img/example.jpg",
        label_rel="train/img/example.txt",
    )

    vm = infer_view(tmp_path)

    assert vm.predicted_format == "yolo"


def test_convert_view_passes_validation_mode_to_conversion_request(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}

    def _fake_execute_conversion(request, *, progress_callback=None):  # type: ignore[no-untyped-def]
        del progress_callback
        observed["validation_mode"] = request.validation_mode.value
        observed["permissive_invalid_annotation_action"] = request.permissive_invalid_annotation_action.value
        observed["allow_overwrite"] = request.allow_overwrite
        observed["input_path_include_substring"] = request.input_path_include_substring
        observed["input_path_exclude_substring"] = request.input_path_exclude_substring
        observed["custom_format_id"] = request.custom_format_id
        observed["custom_format_path"] = request.custom_format_path
        return SimpleNamespace(
            report=SimpleNamespace(
                summary_counts=SimpleNamespace(
                    annotations_in=1,
                    annotations_out=1,
                    dropped=0,
                    unmapped=0,
                ),
                contention_events=[],
            ),
        )

    monkeypatch.setattr(gui_viewmodels, "execute_conversion", _fake_execute_conversion)

    vm, _result = convert_view(
        input_path=tmp_path,
        output_path=tmp_path / "out",
        src="coco",
        dst="yolo",
        custom_format_id="bdd100k_detection",
        custom_format_path=tmp_path / "custom_format.yaml",
        map_path=None,
        unmapped_policy="error",
        dry_run=True,
        allow_overwrite=True,
        input_path_include_substring="train",
        input_path_exclude_substring="backup",
        validation_mode="permissive",
        permissive_invalid_annotation_action="drop",
    )

    assert observed["validation_mode"] == "permissive"
    assert observed["permissive_invalid_annotation_action"] == "drop"
    assert observed["allow_overwrite"] is True
    assert observed["input_path_include_substring"] == "train"
    assert observed["input_path_exclude_substring"] == "backup"
    assert observed["custom_format_id"] == "bdd100k_detection"
    assert observed["custom_format_path"] == tmp_path / "custom_format.yaml"
    assert vm.annotations_in == 1


def test_preview_dataset_view_passes_custom_format_id_to_validation(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}

    def _fake_validate_dataset(  # type: ignore[no-untyped-def]
        input_path,
        *,
        source_format,
        policy,
        load_progress_callback=None,
        annotation_progress_callback=None,
        input_path_include_substring=None,
        input_path_exclude_substring=None,
        custom_format_id=None,
        custom_format_path=None,
    ):
        del input_path, policy, load_progress_callback, annotation_progress_callback
        del input_path_include_substring, input_path_exclude_substring
        observed["source_format"] = source_format
        observed["custom_format_id"] = custom_format_id
        observed["custom_format_path"] = custom_format_path
        return SimpleNamespace(
            dataset=SimpleNamespace(images=[], annotations=[], categories={}),
            warnings=[],
            summary=SimpleNamespace(invalid_annotations=0, errors=[]),
        )

    monkeypatch.setattr(gui_viewmodels, "validate_dataset", _fake_validate_dataset)
    _preview_dataset_view_cached.cache_clear()

    preview = preview_dataset_view(
        tmp_path,
        source_format="custom",
        custom_format_id="bdd100k_detection",
        custom_format_path=tmp_path / "custom_format.yaml",
    )

    assert preview.source_format == "custom"
    assert observed["source_format"] == SourceFormat.CUSTOM
    assert observed["custom_format_id"] == "bdd100k_detection"
    assert observed["custom_format_path"] == tmp_path / "custom_format.yaml"


def test_preview_dataset_view_custom_honors_preview_scan_limit(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}
    sampled_dataset = SimpleNamespace(
        images=[],
        annotations=[],
        categories={},
        source_metadata=SimpleNamespace(details={"records_loaded": "2", "records_total": "10"}),
        warnings=[],
    )

    def _fake_read_custom_dataset(  # type: ignore[no-untyped-def]
        input_path,
        *,
        format_id=None,
        format_path=None,
        input_path_filter=None,
        max_records=None,
    ):
        del input_path, input_path_filter
        observed["format_id"] = format_id
        observed["format_path"] = format_path
        observed["max_records"] = max_records
        return sampled_dataset

    def _fake_validate_loaded_dataset(dataset, *, source_format, policy, annotation_progress_callback=None):  # type: ignore[no-untyped-def]
        del policy, annotation_progress_callback
        observed["validated_dataset"] = dataset
        observed["validated_source_format"] = source_format
        return SimpleNamespace(
            dataset=dataset,
            warnings=[],
            summary=SimpleNamespace(invalid_annotations=0, errors=[]),
        )

    monkeypatch.setattr(gui_viewmodels, "read_custom_dataset", _fake_read_custom_dataset)
    monkeypatch.setattr(gui_viewmodels, "validate_loaded_dataset", _fake_validate_loaded_dataset)
    _preview_dataset_view_cached.cache_clear()

    preview = preview_dataset_view(
        tmp_path,
        source_format="custom",
        custom_format_id="bdd100k_detection",
        custom_format_path=tmp_path / "custom_format.yaml",
        preview_scan_limit=2,
    )

    assert observed["format_id"] == "bdd100k_detection"
    assert observed["format_path"] == tmp_path / "custom_format.yaml"
    assert observed["max_records"] == 2
    assert observed["validated_dataset"] is sampled_dataset
    assert observed["validated_source_format"] == SourceFormat.CUSTOM
    assert preview.warnings == [
        "Custom preview scanned a limited subset (2 / 10 records, limit: 2). Full validation and conversion still process the complete dataset."
    ]

    _preview_dataset_view_cached.cache_clear()


def test_build_gui_run_config_includes_validation_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = build_gui_run_config(
        run_id="gui-test",
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        src="coco",
        dst="yolo",
        custom_format_id="bdd100k_detection",
        custom_format_path=tmp_path / "custom_format.yaml",
        map_path=None,
        unmapped_policy="error",
        dry_run=False,
        allow_overwrite=True,
        input_path_include_substring="train",
        input_path_exclude_substring="backup",
        validation_mode="permissive",
        permissive_invalid_annotation_action="drop",
    )

    assert config.validation_mode == "permissive"
    assert config.permissive_invalid_annotation_action == "drop"
    assert config.allow_overwrite is True
    assert config.custom_format_id == "bdd100k_detection"
    assert config.custom_format_path == str(tmp_path / "custom_format.yaml")
    assert config.input_path_include_substring == "train"
    assert config.input_path_exclude_substring == "backup"
