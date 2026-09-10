from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image

from label_master.adapters.custom.reader import read_custom_dataset
from label_master.adapters.matlab_ground_truth.reader import read_matlab_ground_truth_dataset
from label_master.adapters.video_bbox.reader import read_video_bbox_dataset
from label_master.adapters.voc.reader import read_voc_dataset
from label_master.adapters.yolo.reader import read_yolo_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
    DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    InferencePolicy,
    InvalidAnnotationAction,
    OutOfFrameBBoxPolicy,
    OversizeImageAction,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.domain.value_objects import BBoxCXCYWHNormalized
from label_master.core.services.convert_service import (
    ConversionProgressCallback,
    ConvertRequest,
    ConvertResult,
    execute_conversion,
)
from label_master.core.services.infer_service import infer_format
from label_master.core.services.missing_label_hint_service import (
    HintDetection,
    SingleImageYoloDetectorReview,
    YoloBBoxAuditItem,
    YoloBBoxAuditProposal,
    apply_yolo_bbox_audit,
    discover_yolo_review_image_paths,
    generate_missing_yolo_label_hints,
    generate_yolo_bbox_audit,
    generate_yolo_detector_review,
    generate_yolo_detector_review_for_image,
    load_existing_yolo_labels_for_rel_path,
    load_yolo_class_names_for_dataset_root,
    write_yolo_hint_label_file,
)
from label_master.core.services.validate_service import validate_dataset, validate_loaded_dataset
from label_master.infra.config import load_mapping_file
from label_master.infra.filesystem import build_input_path_filter, safe_resolve
from label_master.infra.reporting import generate_run_id
from label_master.reports.schemas import RunConfigModel

MATLAB_GROUND_TRUTH_PREVIEW_MAX_ANNOTATION_FILES = 1
MATLAB_GROUND_TRUTH_PREVIEW_WARNING = (
    "MATLAB Ground Truth preview shows the first matched video/annotation pair for responsiveness. "
    "Full validation and conversion still process the complete dataset."
)
VOC_PREVIEW_MAX_XML_FILES = 300
VOC_PREVIEW_WARNING = (
    f"Pascal VOC preview shows a capped sample of up to {VOC_PREVIEW_MAX_XML_FILES} annotation/image pairs "
    "for responsiveness. Full validation and conversion still process the complete dataset."
)
VIDEO_BBOX_PREVIEW_MAX_SOURCES = 2
VIDEO_BBOX_PREVIEW_WARNING = (
    f"Video preview shows a capped sample of up to {VIDEO_BBOX_PREVIEW_MAX_SOURCES} source video(s) "
    "for responsiveness. Full validation and conversion still process the complete dataset."
)
ANNOTATION_OVERLAY_COLOR = "blue"
DETECTOR_OVERLAY_COLOR = "orange"

BBoxReviewStrategy = Literal["detector", "smallest", "closest_bounds"]


@dataclass(frozen=True)
class InferenceViewModel:
    predicted_format: str
    confidence: float
    candidates: list[tuple[str, float]]
    warnings: list[str]


@dataclass(frozen=True)
class ConversionViewModel:
    run_id: str
    annotations_in: int
    annotations_out: int
    dropped: int
    unmapped: int
    contention_events: int


@dataclass(frozen=True)
class MissingLabelHintsViewModel:
    scanned_images: int
    images_with_existing_labels: int
    missing_label_images: int
    hinted_images: int
    hint_files_written: int
    total_detections: int
    hints_output_dir: str
    report_path: str
    sample_hints: list[dict[str, str | int]]
    hints: list["MissingLabelHintItemViewModel"]


@dataclass(frozen=True)
class MissingLabelHintDetectionViewModel:
    class_id: int
    class_name: str
    confidence: float
    bbox_xywh_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class MissingLabelHintItemViewModel:
    image_rel_path: str
    suggested_label_rel_path: str
    detections: list[MissingLabelHintDetectionViewModel]


@dataclass(frozen=True)
class MissingLabelHintApprovalViewModel:
    approved_label_files: int
    approved_detections: int
    label_paths: list[str]


@dataclass(frozen=True)
class BBoxAuditProposalViewModel:
    proposal_id: str
    action: str
    class_id: int
    class_name: str
    confidence: float | None
    match_iou: float | None
    existing_label_index: int | None
    existing_bbox_xywh_normalized: tuple[float, float, float, float] | None
    proposed_bbox_xywh_normalized: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class BBoxAuditItemViewModel:
    image_rel_path: str
    label_rel_path: str
    existing_label_count: int
    proposals: list[BBoxAuditProposalViewModel]


@dataclass(frozen=True)
class BBoxAuditResultViewModel:
    scanned_labeled_images: int
    images_with_proposals: int
    add_proposals: int
    remove_proposals: int
    adjust_proposals: int
    report_output_dir: str
    report_path: str
    items: list[BBoxAuditItemViewModel]


@dataclass(frozen=True)
class DetectorReviewItemViewModel:
    source_kind: str
    image_rel_path: str
    label_rel_path: str
    existing_label_count: int
    proposals: list[BBoxAuditProposalViewModel]


@dataclass(frozen=True)
class DetectorReviewResultViewModel:
    missing_label_hints: MissingLabelHintsViewModel
    bbox_audit: BBoxAuditResultViewModel
    items: list[DetectorReviewItemViewModel]


@dataclass(frozen=True)
class BBoxAuditApprovalViewModel:
    approved_label_files: int
    applied_proposals: int
    label_paths: list[str]


@dataclass(frozen=True)
class DetectorReviewApprovalViewModel:
    approved_label_files: int
    applied_proposals: int
    label_paths: list[str]


@dataclass(frozen=True)
class DetectorReviewClassOptionViewModel:
    class_id: int
    class_name: str


@dataclass(frozen=True)
class DetectorReviewEditorBoxViewModel:
    box_id: str
    class_id: int
    class_name: str
    bbox_xywh_normalized: tuple[float, float, float, float]
    source: str


@dataclass(frozen=True)
class DetectorReviewEditorStateViewModel:
    annotation_boxes: list[DetectorReviewEditorBoxViewModel]
    detector_boxes: list[DetectorReviewEditorBoxViewModel]
    editable_boxes: list[DetectorReviewEditorBoxViewModel]
    class_options: list[DetectorReviewClassOptionViewModel]


def _bbox_area(
    bbox_xywh_normalized: tuple[float, float, float, float],
) -> float:
    return max(bbox_xywh_normalized[2], 0.0) * max(bbox_xywh_normalized[3], 0.0)


def _bbox_to_xyxy(
    bbox_xywh_normalized: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = bbox_xywh_normalized
    half_width = max(width, 0.0) / 2.0
    half_height = max(height, 0.0) / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _bbox_from_xyxy(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float, float] | None:
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        left + (width / 2.0),
        top + (height / 2.0),
        width,
        height,
    )


def _bbox_is_same(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    return all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )


def _resolve_adjusted_bbox(
    *,
    existing_bbox_xywh_normalized: tuple[float, float, float, float],
    proposed_bbox_xywh_normalized: tuple[float, float, float, float],
    strategy: BBoxReviewStrategy,
) -> tuple[float, float, float, float]:
    if strategy == "detector":
        return proposed_bbox_xywh_normalized

    if strategy == "smallest":
        if _bbox_area(existing_bbox_xywh_normalized) <= _bbox_area(proposed_bbox_xywh_normalized):
            return existing_bbox_xywh_normalized
        return proposed_bbox_xywh_normalized

    existing_left, existing_top, existing_right, existing_bottom = _bbox_to_xyxy(
        existing_bbox_xywh_normalized,
    )
    proposed_left, proposed_top, proposed_right, proposed_bottom = _bbox_to_xyxy(
        proposed_bbox_xywh_normalized,
    )
    intersection_bbox = _bbox_from_xyxy(
        max(existing_left, proposed_left),
        max(existing_top, proposed_top),
        min(existing_right, proposed_right),
        min(existing_bottom, proposed_bottom),
    )
    if intersection_bbox is not None:
        return intersection_bbox

    if _bbox_area(existing_bbox_xywh_normalized) <= _bbox_area(proposed_bbox_xywh_normalized):
        return existing_bbox_xywh_normalized
    return proposed_bbox_xywh_normalized


def detector_review_final_bbox_xywh_normalized(
    proposal: BBoxAuditProposalViewModel,
    *,
    strategy: BBoxReviewStrategy,
) -> tuple[float, float, float, float] | None:
    if proposal.action == "remove":
        return None

    if proposal.action == "add":
        return proposal.proposed_bbox_xywh_normalized

    existing_bbox = proposal.existing_bbox_xywh_normalized
    proposed_bbox = proposal.proposed_bbox_xywh_normalized
    if existing_bbox is None or proposed_bbox is None:
        return proposed_bbox

    return _resolve_adjusted_bbox(
        existing_bbox_xywh_normalized=existing_bbox,
        proposed_bbox_xywh_normalized=proposed_bbox,
        strategy=strategy,
    )


def resolve_detector_review_proposal(
    proposal: BBoxAuditProposalViewModel,
    *,
    strategy: BBoxReviewStrategy,
) -> BBoxAuditProposalViewModel | None:
    final_bbox = detector_review_final_bbox_xywh_normalized(
        proposal,
        strategy=strategy,
    )
    if proposal.action == "adjust" and final_bbox is not None:
        existing_bbox = proposal.existing_bbox_xywh_normalized
        if existing_bbox is not None and _bbox_is_same(final_bbox, existing_bbox):
            return None
        return replace(
            proposal,
            proposed_bbox_xywh_normalized=final_bbox,
        )

    return proposal


def apply_detector_review_bbox_strategy(
    proposals: list[BBoxAuditProposalViewModel],
    *,
    strategy: BBoxReviewStrategy,
) -> list[BBoxAuditProposalViewModel]:
    transformed_proposals: list[BBoxAuditProposalViewModel] = []
    for proposal in proposals:
        resolved_proposal = resolve_detector_review_proposal(
            proposal,
            strategy=strategy,
        )
        if resolved_proposal is None:
            continue
        transformed_proposals.append(resolved_proposal)

    return transformed_proposals


def _annotation_box_id(label_index: int) -> str:
    return f"annotation:{label_index}"


def _detector_box_id(proposal_id: str) -> str:
    return f"detector:{proposal_id}"


def _editable_box_id(proposal_id: str) -> str:
    return f"editable:{proposal_id}"


def _existing_label_to_editor_box(
    *,
    label_index: int,
    class_id: int,
    class_name: str,
    bbox_xywh_normalized: tuple[float, float, float, float],
    source: str,
) -> DetectorReviewEditorBoxViewModel:
    return DetectorReviewEditorBoxViewModel(
        box_id=_annotation_box_id(label_index),
        class_id=class_id,
        class_name=class_name,
        bbox_xywh_normalized=bbox_xywh_normalized,
        source=source,
    )


def _proposal_to_detector_editor_box(
    proposal: BBoxAuditProposalViewModel,
) -> DetectorReviewEditorBoxViewModel | None:
    if proposal.proposed_bbox_xywh_normalized is None:
        return None
    return DetectorReviewEditorBoxViewModel(
        box_id=_detector_box_id(proposal.proposal_id),
        class_id=proposal.class_id,
        class_name=proposal.class_name,
        bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
        source="detector",
    )


def _class_options_from_editor_boxes(
    boxes: list[DetectorReviewEditorBoxViewModel],
    proposals: list[BBoxAuditProposalViewModel],
) -> list[DetectorReviewClassOptionViewModel]:
    class_names: dict[int, str] = {}
    for box in boxes:
        class_names[box.class_id] = box.class_name
    for proposal in proposals:
        class_names[proposal.class_id] = proposal.class_name
    if not class_names:
        class_names[0] = "class_0"
    return [
        DetectorReviewClassOptionViewModel(class_id=class_id, class_name=class_name)
        for class_id, class_name in sorted(class_names.items())
    ]


def build_detector_review_editor_state_view(
    *,
    dataset_root: Path,
    review_item: DetectorReviewItemViewModel,
    selected_proposals: list[BBoxAuditProposalViewModel],
    strategy: BBoxReviewStrategy,
) -> DetectorReviewEditorStateViewModel:
    existing_labels = load_existing_yolo_labels_for_rel_path(
        dataset_root=dataset_root,
        label_rel_path=review_item.label_rel_path,
    )
    annotation_boxes = [
        _existing_label_to_editor_box(
            label_index=label.label_index,
            class_id=label.class_id,
            class_name=label.class_name,
            bbox_xywh_normalized=label.bbox_xywh_normalized,
            source="annotation",
        )
        for label in existing_labels
    ]
    detector_boxes = [
        box
        for box in (_proposal_to_detector_editor_box(proposal) for proposal in review_item.proposals)
        if box is not None
    ]

    editable_by_id: dict[str, DetectorReviewEditorBoxViewModel] = {
        box.box_id: replace(box, source="editable")
        for box in annotation_boxes
    }
    editable_order = [box.box_id for box in annotation_boxes]

    for proposal in selected_proposals:
        resolved_proposal = resolve_detector_review_proposal(
            proposal,
            strategy=strategy,
        )
        if resolved_proposal is None:
            continue

        if resolved_proposal.action == "remove" and resolved_proposal.existing_label_index is not None:
            box_id = _annotation_box_id(resolved_proposal.existing_label_index)
            editable_by_id.pop(box_id, None)
            editable_order = [candidate_id for candidate_id in editable_order if candidate_id != box_id]
            continue

        if (
            resolved_proposal.action == "adjust"
            and resolved_proposal.existing_label_index is not None
            and resolved_proposal.proposed_bbox_xywh_normalized is not None
        ):
            box_id = _annotation_box_id(resolved_proposal.existing_label_index)
            editable_by_id[box_id] = DetectorReviewEditorBoxViewModel(
                box_id=box_id,
                class_id=resolved_proposal.class_id,
                class_name=resolved_proposal.class_name,
                bbox_xywh_normalized=resolved_proposal.proposed_bbox_xywh_normalized,
                source="editable",
            )
            if box_id not in editable_order:
                editable_order.append(box_id)
            continue

        if resolved_proposal.action == "add" and resolved_proposal.proposed_bbox_xywh_normalized is not None:
            box_id = _editable_box_id(resolved_proposal.proposal_id)
            editable_by_id[box_id] = DetectorReviewEditorBoxViewModel(
                box_id=box_id,
                class_id=resolved_proposal.class_id,
                class_name=resolved_proposal.class_name,
                bbox_xywh_normalized=resolved_proposal.proposed_bbox_xywh_normalized,
                source="editable",
            )
            if box_id not in editable_order:
                editable_order.append(box_id)

    try:
        class_name_map = load_yolo_class_names_for_dataset_root(dataset_root)
    except FileNotFoundError:
        class_name_map = {}
    class_options = (
        [
            DetectorReviewClassOptionViewModel(class_id=class_id, class_name=class_name)
            for class_id, class_name in sorted(class_name_map.items())
        ]
        if class_name_map
        else _class_options_from_editor_boxes(annotation_boxes + detector_boxes, review_item.proposals)
    )

    return DetectorReviewEditorStateViewModel(
        annotation_boxes=annotation_boxes,
        detector_boxes=detector_boxes,
        editable_boxes=[editable_by_id[box_id] for box_id in editable_order if box_id in editable_by_id],
        class_options=class_options,
    )


@dataclass(frozen=True)
class PreviewBBoxViewModel:
    annotation_id: str
    class_id: int
    class_name: str
    bbox_xywh_abs: tuple[float, float, float, float]


@dataclass(frozen=True)
class PreviewImageViewModel:
    image_id: str
    file_name: str
    width: int
    height: int
    bboxes: list[PreviewBBoxViewModel]


@dataclass(frozen=True)
class DatasetPreviewViewModel:
    source_format: str
    image_count: int
    images: list[PreviewImageViewModel]
    warnings: list[str]


@dataclass(frozen=True)
class MappingRowViewModel:
    source_class_id: str
    action: str
    destination_class_id: str


@dataclass(frozen=True)
class MappingParseViewModel:
    class_map: dict[int, int | None]
    drop_frames_with_class_ids: frozenset[int]
    errors: list[str]


def _preview_invalid_annotation_warning(
    invalid_annotations: int,
    errors: list[str],
    *,
    out_of_frame_bbox_policy: str,
    out_of_frame_tolerance_px: float,
) -> str | None:
    if invalid_annotations <= 0:
        return None

    frame_overflow_errors = sum("bbox goes out of frame" in error for error in errors)
    if frame_overflow_errors == invalid_annotations:
        if out_of_frame_bbox_policy not in {"correct", "warn"}:
            return (
                f"Preview loaded with {invalid_annotations} invalid annotation(s): "
                "bbox goes out of frame. Enable 'Correct' or 'Warn' bbox policy in Step 3 to clip near-edge boxes."
            )
        return (
            f"Preview loaded with {invalid_annotations} invalid annotation(s): "
            f"bbox goes out of frame beyond the accepted {out_of_frame_tolerance_px:g}px correction tolerance."
        )
    if frame_overflow_errors:
        if out_of_frame_bbox_policy not in {"correct", "warn"}:
            return (
                f"Preview loaded with {invalid_annotations} invalid annotation(s), including "
                f"{frame_overflow_errors} bbox(es) that go out of frame."
            )
        return (
            f"Preview loaded with {invalid_annotations} invalid annotation(s), including "
            f"{frame_overflow_errors} bbox(es) that go out of frame beyond the accepted "
            f"{out_of_frame_tolerance_px:g}px correction tolerance."
        )
    return f"Preview loaded with {invalid_annotations} invalid annotation(s)."


@lru_cache(maxsize=16)
def _preview_dataset_view_cached(
    input_path: str,
    source_format: str,
    out_of_frame_bbox_policy: str,
    out_of_frame_tolerance_px: float,
    input_path_include_substring: str | None,
    input_path_exclude_substring: str | None,
    preview_scan_limit: int,
    custom_format_id: str | None,
    custom_format_path: str | None,
) -> DatasetPreviewViewModel:
    resolved_format = SourceFormat(source_format)
    if resolved_format not in {
        SourceFormat.CITYSCAPES,
        SourceFormat.COCO,
        SourceFormat.CUSTOM,
        SourceFormat.KITWARE,
        SourceFormat.MATLAB_GROUND_TRUTH,
        SourceFormat.VOC,
        SourceFormat.VIDEO_BBOX,
        SourceFormat.YOLO,
    }:
        raise ValueError(
            "Preview source format must be cityscapes, coco, custom, kitware, matlab_ground_truth, voc, video_bbox, or yolo"
        )

    preview_policy = ValidationPolicy.for_mode(
        ValidationMode.PERMISSIVE,
        out_of_frame_bbox_policy=OutOfFrameBBoxPolicy(out_of_frame_bbox_policy),
        out_of_frame_tolerance_px=out_of_frame_tolerance_px,
    )
    effective_preview_scan_limit = preview_scan_limit if preview_scan_limit > 0 else None
    warnings: list[str] = []
    if resolved_format == SourceFormat.MATLAB_GROUND_TRUTH:
        annotation_files_limit = (
            effective_preview_scan_limit
            if effective_preview_scan_limit is not None
            else MATLAB_GROUND_TRUTH_PREVIEW_MAX_ANNOTATION_FILES
        )
        sampled_dataset = read_matlab_ground_truth_dataset(
            Path(input_path),
            max_annotation_files=annotation_files_limit,
            input_path_filter=build_input_path_filter(
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            ),
        )
        validation = validate_loaded_dataset(
            sampled_dataset,
            source_format=resolved_format,
            policy=preview_policy,
        )
        if effective_preview_scan_limit is not None:
            warnings.append(
                "Preview scan limit is active and may sample only part of MATLAB Ground Truth annotations. "
                "Full validation and conversion still process the complete dataset."
            )
        else:
            warnings.append(MATLAB_GROUND_TRUTH_PREVIEW_WARNING)
    elif resolved_format == SourceFormat.VOC:
        xml_files_limit = (
            effective_preview_scan_limit
            if effective_preview_scan_limit is not None
            else VOC_PREVIEW_MAX_XML_FILES
        )
        sampled_dataset = read_voc_dataset(
            Path(input_path),
            max_xml_files=xml_files_limit,
            input_path_filter=build_input_path_filter(
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            ),
        )
        validation = validate_loaded_dataset(
            sampled_dataset,
            source_format=resolved_format,
            policy=preview_policy,
        )
        xml_files_loaded = int(sampled_dataset.source_metadata.details.get("xml_files_loaded", "0"))
        xml_files_total = int(sampled_dataset.source_metadata.details.get("xml_files_total", str(xml_files_loaded)))
        if xml_files_loaded < xml_files_total:
            if effective_preview_scan_limit is not None:
                warnings.append(
                    f"VOC preview scanned a limited subset ({xml_files_loaded} / {xml_files_total} XML files, "
                    f"limit: {xml_files_limit}). Full validation and conversion still process the complete dataset."
                )
            else:
                warnings.append(VOC_PREVIEW_WARNING)
    elif resolved_format == SourceFormat.VIDEO_BBOX:
        sources_limit = (
            effective_preview_scan_limit
            if effective_preview_scan_limit is not None
            else VIDEO_BBOX_PREVIEW_MAX_SOURCES
        )
        sampled_dataset = read_video_bbox_dataset(
            Path(input_path),
            max_sources=sources_limit,
            input_path_filter=build_input_path_filter(
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            ),
        )
        validation = validate_loaded_dataset(
            sampled_dataset,
            source_format=resolved_format,
            policy=preview_policy,
        )
        video_sources_loaded = int(
            sampled_dataset.source_metadata.details.get("video_sources_loaded", "0")
        )
        video_sources_total = int(
            sampled_dataset.source_metadata.details.get(
                "video_sources_total",
                str(video_sources_loaded),
            )
        )
        if video_sources_loaded < video_sources_total:
            if effective_preview_scan_limit is not None:
                warnings.append(
                    f"Video preview scanned a limited subset ({video_sources_loaded} / {video_sources_total} "
                    f"sources, limit: {sources_limit}). Full validation and conversion still process the complete dataset."
                )
            else:
                warnings.append(VIDEO_BBOX_PREVIEW_WARNING)
    elif resolved_format == SourceFormat.YOLO and effective_preview_scan_limit is not None:
        sampled_dataset = read_yolo_dataset(
            Path(input_path),
            max_label_files=effective_preview_scan_limit,
            input_path_filter=build_input_path_filter(
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            ),
        )
        validation = validate_loaded_dataset(
            sampled_dataset,
            source_format=resolved_format,
            policy=preview_policy,
        )
        label_files_loaded = int(sampled_dataset.source_metadata.details.get("label_files_loaded", "0"))
        label_files_total = int(
            sampled_dataset.source_metadata.details.get(
                "label_files_total",
                str(label_files_loaded),
            )
        )
        if label_files_loaded < label_files_total:
            warnings.append(
                f"YOLO preview scanned a limited subset ({label_files_loaded} / {label_files_total} label files, "
                f"limit: {effective_preview_scan_limit}). Full validation and conversion still process the complete dataset."
            )
    elif resolved_format == SourceFormat.CUSTOM and effective_preview_scan_limit is not None:
        sampled_dataset = read_custom_dataset(
            Path(input_path),
            format_id=custom_format_id,
            format_path=Path(custom_format_path) if custom_format_path else None,
            input_path_filter=build_input_path_filter(
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            ),
            max_records=effective_preview_scan_limit,
        )
        validation = validate_loaded_dataset(
            sampled_dataset,
            source_format=resolved_format,
            policy=preview_policy,
        )
        records_loaded = int(sampled_dataset.source_metadata.details.get("records_loaded", "0"))
        records_total = int(
            sampled_dataset.source_metadata.details.get(
                "records_total",
                str(records_loaded),
            )
        )
        if records_loaded < records_total:
            warnings.append(
                f"Custom preview scanned a limited subset ({records_loaded} / {records_total} records, "
                f"limit: {effective_preview_scan_limit}). Full validation and conversion still process the complete dataset."
            )
    else:
        validation = validate_dataset(
            Path(input_path),
            source_format=resolved_format,
            policy=preview_policy,
            input_path_include_substring=input_path_include_substring,
            input_path_exclude_substring=input_path_exclude_substring,
            custom_format_id=custom_format_id,
            custom_format_path=Path(custom_format_path) if custom_format_path else None,
        )
    dataset = validation.dataset

    warnings.extend(
        str(warning.message)
        for warning in getattr(validation, "warnings", [])
        if getattr(warning, "message", "")
    )
    invalid_warning = _preview_invalid_annotation_warning(
        int(getattr(validation.summary, "invalid_annotations", 0)),
        [str(error) for error in getattr(validation.summary, "errors", [])],
        out_of_frame_bbox_policy=out_of_frame_bbox_policy,
        out_of_frame_tolerance_px=out_of_frame_tolerance_px,
    )
    if invalid_warning:
        warnings.append(invalid_warning)

    bboxes_by_image: dict[str, list[PreviewBBoxViewModel]] = defaultdict(list)
    for annotation in dataset.annotations:
        category = dataset.categories.get(annotation.class_id)
        bboxes_by_image[annotation.image_id].append(
            PreviewBBoxViewModel(
                annotation_id=annotation.annotation_id,
                class_id=annotation.class_id,
                class_name=category.name if category else f"class_{annotation.class_id}",
                bbox_xywh_abs=annotation.bbox_xywh_abs,
            )
        )

    images = [
        PreviewImageViewModel(
            image_id=image.image_id,
            file_name=image.file_name,
            width=image.width,
            height=image.height,
            bboxes=bboxes_by_image.get(image.image_id, []),
        )
        for image in dataset.images
    ]

    return DatasetPreviewViewModel(
        source_format=resolved_format.value,
        image_count=len(images),
        images=images,
        warnings=warnings,
    )


def preview_dataset_view(
    input_path: Path,
    *,
    source_format: str,
    out_of_frame_bbox_policy: str = DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
    out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    preview_scan_limit: int = 0,
    custom_format_id: str | None = None,
    custom_format_path: Path | None = None,
) -> DatasetPreviewViewModel:
    normalized_preview_scan_limit = max(0, int(preview_scan_limit))
    return _preview_dataset_view_cached(
        str(input_path.expanduser().resolve()),
        source_format,
        out_of_frame_bbox_policy,
        float(out_of_frame_tolerance_px),
        input_path_include_substring,
        input_path_exclude_substring,
        normalized_preview_scan_limit,
        custom_format_id,
        str(custom_format_path.expanduser().resolve()) if custom_format_path else None,
    )


def parse_mapping_rows(rows: list[MappingRowViewModel]) -> MappingParseViewModel:
    class_map: dict[int, int | None] = {}
    drop_frame_ids: set[int] = set()
    errors: list[str] = []
    seen_source_ids: set[int] = set()

    for index, row in enumerate(rows, start=1):
        source_raw = row.source_class_id.strip()
        action = row.action.strip().lower() or "map"
        destination_raw = row.destination_class_id.strip()

        if not source_raw and not destination_raw:
            continue

        if not source_raw:
            errors.append(f"Row {index}: source_class_id is required")
            continue

        try:
            source_class_id = int(source_raw)
        except ValueError:
            errors.append(f"Row {index}: source_class_id must be an integer")
            continue

        if source_class_id in seen_source_ids:
            errors.append(f"Row {index}: duplicate source_class_id {source_class_id}")
            continue
        seen_source_ids.add(source_class_id)

        if action not in {"map", "drop", "drop_frame"}:
            errors.append(f"Row {index}: action must be 'map', 'drop', or 'drop_frame'")
            continue

        if action == "drop":
            class_map[source_class_id] = None
            continue

        if action == "drop_frame":
            drop_frame_ids.add(source_class_id)
            continue

        if not destination_raw:
            errors.append(f"Row {index}: destination_class_id is required when action is 'map'")
            continue

        try:
            destination_class_id = int(destination_raw)
        except ValueError:
            errors.append(f"Row {index}: destination_class_id must be an integer")
            continue

        class_map[source_class_id] = destination_class_id

    return MappingParseViewModel(
        class_map=class_map,
        drop_frames_with_class_ids=frozenset(drop_frame_ids),
        errors=errors,
    )


def infer_view(input_path: Path, *, sample_limit: int = 100) -> InferenceViewModel:
    result = infer_format(input_path, policy=InferencePolicy(sample_limit=sample_limit), force=True)
    return InferenceViewModel(
        predicted_format=result.predicted_format.value,
        confidence=result.confidence,
        candidates=[(candidate.format.value, candidate.score) for candidate in result.candidates],
        warnings=[warning.message for warning in result.warnings],
    )


def _missing_label_hints_result_to_viewmodel(result) -> MissingLabelHintsViewModel:  # type: ignore[no-untyped-def]
    sample_hints = [
        {
            "image": hint.image_rel_path,
            "suggested_label": hint.suggested_label_rel_path,
            "detections": len(hint.detections),
        }
        for hint in result.hints[:50]
    ]
    hints = [
        MissingLabelHintItemViewModel(
            image_rel_path=hint.image_rel_path,
            suggested_label_rel_path=hint.suggested_label_rel_path,
            detections=[
                MissingLabelHintDetectionViewModel(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox_xywh_normalized=detection.bbox_xywh_normalized,
                )
                for detection in hint.detections
            ],
        )
        for hint in result.hints
    ]
    return MissingLabelHintsViewModel(
        scanned_images=result.scanned_images,
        images_with_existing_labels=result.images_with_existing_labels,
        missing_label_images=result.missing_label_images,
        hinted_images=result.hinted_images,
        hint_files_written=result.hint_files_written,
        total_detections=result.total_detections,
        hints_output_dir=str(result.hints_output_dir),
        report_path=str(result.report_path),
        sample_hints=sample_hints,
        hints=hints,
    )


def _bbox_audit_result_to_viewmodel(result) -> BBoxAuditResultViewModel:  # type: ignore[no-untyped-def]
    items = [
        BBoxAuditItemViewModel(
            image_rel_path=item.image_rel_path,
            label_rel_path=item.label_rel_path,
            existing_label_count=item.existing_label_count,
            proposals=[
                BBoxAuditProposalViewModel(
                    proposal_id=proposal.proposal_id,
                    action=proposal.action,
                    class_id=proposal.class_id,
                    class_name=proposal.class_name,
                    confidence=proposal.confidence,
                    match_iou=proposal.match_iou,
                    existing_label_index=proposal.existing_label_index,
                    existing_bbox_xywh_normalized=proposal.existing_bbox_xywh_normalized,
                    proposed_bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
                )
                for proposal in item.proposals
            ],
        )
        for item in result.items
    ]

    return BBoxAuditResultViewModel(
        scanned_labeled_images=result.scanned_labeled_images,
        images_with_proposals=result.images_with_proposals,
        add_proposals=result.add_proposals,
        remove_proposals=result.remove_proposals,
        adjust_proposals=result.adjust_proposals,
        report_output_dir=str(result.report_output_dir),
        report_path=str(result.report_path),
        items=items,
    )


def generate_missing_label_hints_view(
    *,
    input_path: Path,
    source_format: str,
    detector_model_path: Path,
    hints_output_dir: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> MissingLabelHintsViewModel:
    resolved_source_format = SourceFormat(source_format)
    if resolved_source_format != SourceFormat.YOLO:
        raise ValueError("Missing-label detector hints currently support YOLO source datasets only.")

    result = generate_missing_yolo_label_hints(
        dataset_root=input_path,
        detector_model_path=detector_model_path,
        hints_output_dir=hints_output_dir,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
    )
    return _missing_label_hints_result_to_viewmodel(result)


def build_missing_label_hint_overlay_labels(
    *,
    dataset_root: Path,
    hint: MissingLabelHintItemViewModel,
) -> list[tuple[float, float, float, float, str]]:
    image_path = safe_resolve(dataset_root.expanduser().resolve(), hint.image_rel_path)
    with Image.open(image_path) as image:
        width, height = image.size

    overlay_labels: list[tuple[float, float, float, float, str]] = []
    for detection in hint.detections:
        absolute = BBoxCXCYWHNormalized(*detection.bbox_xywh_normalized).to_absolute(width, height)
        overlay_labels.append(
            (
                absolute.x,
                absolute.y,
                absolute.w,
                absolute.h,
                f"{detection.class_id}:{detection.class_name} {detection.confidence:.2f}",
            )
        )
    return overlay_labels


def approve_missing_label_hints_view(
    *,
    dataset_root: Path,
    approved_hints: list[MissingLabelHintItemViewModel],
    allow_overwrite: bool = False,
) -> MissingLabelHintApprovalViewModel:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    label_paths: list[str] = []
    approved_detections = 0

    for hint in approved_hints:
        if not hint.detections:
            continue

        target_label_path = safe_resolve(resolved_root, hint.suggested_label_rel_path)
        if target_label_path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing label file: {target_label_path}")

        detections = [
            HintDetection(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox_xywh_normalized=detection.bbox_xywh_normalized,
            )
            for detection in hint.detections
        ]
        write_yolo_hint_label_file(target_label_path, detections)
        label_paths.append(str(target_label_path))
        approved_detections += len(detections)

    return MissingLabelHintApprovalViewModel(
        approved_label_files=len(label_paths),
        approved_detections=approved_detections,
        label_paths=label_paths,
    )


def generate_bbox_audit_view(
    *,
    input_path: Path,
    source_format: str,
    detector_model_path: Path,
    report_output_dir: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
    max_labeled_images: int,
    match_iou_threshold: float,
    correction_iou_threshold: float,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> BBoxAuditResultViewModel:
    resolved_source_format = SourceFormat(source_format)
    if resolved_source_format != SourceFormat.YOLO:
        raise ValueError("Detector bbox audit currently supports YOLO source datasets only.")

    result = generate_yolo_bbox_audit(
        dataset_root=input_path,
        detector_model_path=detector_model_path,
        report_output_dir=report_output_dir,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        max_labeled_images=max_labeled_images,
        match_iou_threshold=match_iou_threshold,
        correction_iou_threshold=correction_iou_threshold,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
    )

    return _bbox_audit_result_to_viewmodel(result)


def _missing_label_hint_to_detector_review_item(
    hint: MissingLabelHintItemViewModel,
) -> DetectorReviewItemViewModel:
    proposals = [
        BBoxAuditProposalViewModel(
            proposal_id=f"hint:add:{index}",
            action="add",
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            match_iou=None,
            existing_label_index=None,
            existing_bbox_xywh_normalized=None,
            proposed_bbox_xywh_normalized=detection.bbox_xywh_normalized,
        )
        for index, detection in enumerate(hint.detections)
    ]
    return DetectorReviewItemViewModel(
        source_kind="missing_label",
        image_rel_path=hint.image_rel_path,
        label_rel_path=hint.suggested_label_rel_path,
        existing_label_count=0,
        proposals=proposals,
    )


def _bbox_audit_item_to_detector_review_item(
    item: BBoxAuditItemViewModel,
) -> DetectorReviewItemViewModel:
    return DetectorReviewItemViewModel(
        source_kind="bbox_audit",
        image_rel_path=item.image_rel_path,
        label_rel_path=item.label_rel_path,
        existing_label_count=item.existing_label_count,
        proposals=item.proposals,
    )


def _detector_review_items_from_viewmodels(
    missing_label_hints: MissingLabelHintsViewModel,
    bbox_audit: BBoxAuditResultViewModel,
) -> list[DetectorReviewItemViewModel]:
    items = [
        _missing_label_hint_to_detector_review_item(hint)
        for hint in missing_label_hints.hints
    ]
    items.extend(_bbox_audit_item_to_detector_review_item(item) for item in bbox_audit.items)
    return sorted(
        items,
        key=lambda item: (item.image_rel_path, item.label_rel_path, item.source_kind),
    )


def _single_image_detector_review_to_viewmodel(
    review: SingleImageYoloDetectorReview,
) -> DetectorReviewItemViewModel:
    return DetectorReviewItemViewModel(
        source_kind="bbox_audit" if review.has_existing_label_file else "missing_label",
        image_rel_path=review.image_rel_path,
        label_rel_path=review.label_rel_path,
        existing_label_count=review.existing_label_count,
        proposals=[
            BBoxAuditProposalViewModel(
                proposal_id=proposal.proposal_id,
                action=proposal.action,
                class_id=proposal.class_id,
                class_name=proposal.class_name,
                confidence=proposal.confidence,
                match_iou=proposal.match_iou,
                existing_label_index=proposal.existing_label_index,
                existing_bbox_xywh_normalized=proposal.existing_bbox_xywh_normalized,
                proposed_bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
            )
            for proposal in review.proposals
        ],
    )


def list_detector_review_image_paths_view(
    *,
    input_path: Path,
    source_format: str,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> list[str]:
    resolved_source_format = SourceFormat(source_format)
    if resolved_source_format != SourceFormat.YOLO:
        raise ValueError("Detector review currently supports YOLO source datasets only.")

    return discover_yolo_review_image_paths(
        dataset_root=input_path,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
    )


def generate_detector_review_item_view(
    *,
    input_path: Path,
    source_format: str,
    image_rel_path: str,
    detector_model_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
    match_iou_threshold: float,
    correction_iou_threshold: float,
) -> DetectorReviewItemViewModel:
    resolved_source_format = SourceFormat(source_format)
    if resolved_source_format != SourceFormat.YOLO:
        raise ValueError("Detector review currently supports YOLO source datasets only.")

    review = generate_yolo_detector_review_for_image(
        dataset_root=input_path,
        image_rel_path=image_rel_path,
        detector_model_path=detector_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        match_iou_threshold=match_iou_threshold,
        correction_iou_threshold=correction_iou_threshold,
    )
    return _single_image_detector_review_to_viewmodel(review)


def generate_detector_review_view(
    *,
    input_path: Path,
    source_format: str,
    detector_model_path: Path,
    hints_output_dir: Path,
    report_output_dir: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
    max_labeled_images: int,
    match_iou_threshold: float,
    correction_iou_threshold: float,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> DetectorReviewResultViewModel:
    resolved_source_format = SourceFormat(source_format)
    if resolved_source_format != SourceFormat.YOLO:
        raise ValueError("Detector review currently supports YOLO source datasets only.")

    result = generate_yolo_detector_review(
        dataset_root=input_path,
        detector_model_path=detector_model_path,
        hints_output_dir=hints_output_dir,
        report_output_dir=report_output_dir,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        max_labeled_images=max_labeled_images,
        match_iou_threshold=match_iou_threshold,
        correction_iou_threshold=correction_iou_threshold,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
    )

    missing_label_hints_vm = _missing_label_hints_result_to_viewmodel(result.missing_label_hints)
    bbox_audit_vm = _bbox_audit_result_to_viewmodel(result.bbox_audit)

    return DetectorReviewResultViewModel(
        missing_label_hints=missing_label_hints_vm,
        bbox_audit=bbox_audit_vm,
        items=_detector_review_items_from_viewmodels(missing_label_hints_vm, bbox_audit_vm),
    )


def build_bbox_audit_overlay_labels(
    *,
    dataset_root: Path,
    audit_item: BBoxAuditItemViewModel,
    selected_proposals: list[BBoxAuditProposalViewModel],
) -> list[tuple[float, float, float, float, str, str]]:
    image_path = safe_resolve(dataset_root.expanduser().resolve(), audit_item.image_rel_path)
    with Image.open(image_path) as image:
        width, height = image.size

    overlay_labels: list[tuple[float, float, float, float, str, str]] = []
    existing_labels = load_existing_yolo_labels_for_rel_path(
        dataset_root=dataset_root,
        label_rel_path=audit_item.label_rel_path,
    )
    for existing_label in existing_labels:
        existing_absolute = BBoxCXCYWHNormalized(*existing_label.bbox_xywh_normalized).to_absolute(
            width,
            height,
        )
        overlay_labels.append(
            (
                existing_absolute.x,
                existing_absolute.y,
                existing_absolute.w,
                existing_absolute.h,
                f"ann {existing_label.class_id}:{existing_label.class_name}",
                ANNOTATION_OVERLAY_COLOR,
            )
        )

    for proposal in selected_proposals:
        if proposal.action in {"add", "adjust"} and proposal.proposed_bbox_xywh_normalized is not None:
            proposed_absolute = BBoxCXCYWHNormalized(*proposal.proposed_bbox_xywh_normalized).to_absolute(
                width,
                height,
            )
            confidence_suffix = (
                f" {proposal.confidence:.2f}" if proposal.confidence is not None else ""
            )
            overlay_labels.append(
                (
                    proposed_absolute.x,
                    proposed_absolute.y,
                    proposed_absolute.w,
                    proposed_absolute.h,
                    f"det {proposal.action} {proposal.class_id}:{proposal.class_name}{confidence_suffix}",
                    DETECTOR_OVERLAY_COLOR,
                )
            )
    return overlay_labels


def _detector_review_item_to_bbox_audit_item(
    item: DetectorReviewItemViewModel,
) -> BBoxAuditItemViewModel:
    return BBoxAuditItemViewModel(
        image_rel_path=item.image_rel_path,
        label_rel_path=item.label_rel_path,
        existing_label_count=item.existing_label_count,
        proposals=item.proposals,
    )


def build_detector_review_overlay_labels(
    *,
    dataset_root: Path,
    review_item: DetectorReviewItemViewModel,
    selected_proposals: list[BBoxAuditProposalViewModel],
) -> list[tuple[float, float, float, float, str, str]]:
    return build_bbox_audit_overlay_labels(
        dataset_root=dataset_root,
        audit_item=_detector_review_item_to_bbox_audit_item(review_item),
        selected_proposals=selected_proposals,
    )


def approve_bbox_audit_view(
    *,
    dataset_root: Path,
    approved_items: list[BBoxAuditItemViewModel],
) -> BBoxAuditApprovalViewModel:
    result = apply_yolo_bbox_audit(
        dataset_root=dataset_root,
        approved_items=[
            YoloBBoxAuditItem(
                image_rel_path=item.image_rel_path,
                label_rel_path=item.label_rel_path,
                existing_label_count=item.existing_label_count,
                proposals=[
                    YoloBBoxAuditProposal(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,  # type: ignore[arg-type]
                        class_id=proposal.class_id,
                        class_name=proposal.class_name,
                        confidence=proposal.confidence,
                        match_iou=proposal.match_iou,
                        existing_label_index=proposal.existing_label_index,
                        existing_bbox_xywh_normalized=proposal.existing_bbox_xywh_normalized,
                        proposed_bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
                    )
                    for proposal in item.proposals
                ],
            )
            for item in approved_items
        ],
    )
    return BBoxAuditApprovalViewModel(
        approved_label_files=result.approved_label_files,
        applied_proposals=result.applied_proposals,
        label_paths=result.label_paths,
    )


def approve_detector_review_view(
    *,
    dataset_root: Path,
    approved_items: list[DetectorReviewItemViewModel],
    allow_overwrite_missing_label_files: bool = False,
) -> DetectorReviewApprovalViewModel:
    approved_hints: list[MissingLabelHintItemViewModel] = []
    approved_bbox_items: list[BBoxAuditItemViewModel] = []

    for item in approved_items:
        if item.source_kind == "missing_label":
            detections = [
                MissingLabelHintDetectionViewModel(
                    class_id=proposal.class_id,
                    class_name=proposal.class_name,
                    confidence=proposal.confidence or 1.0,
                    bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
                )
                for proposal in item.proposals
                if proposal.action == "add" and proposal.proposed_bbox_xywh_normalized is not None
            ]
            if detections:
                approved_hints.append(
                    MissingLabelHintItemViewModel(
                        image_rel_path=item.image_rel_path,
                        suggested_label_rel_path=item.label_rel_path,
                        detections=detections,
                    )
                )
            continue

        if item.source_kind == "bbox_audit":
            approved_bbox_items.append(_detector_review_item_to_bbox_audit_item(item))
            continue

        raise ValueError(f"Unsupported detector review item source: {item.source_kind}")

    approved_label_files = 0
    applied_proposals = 0
    label_paths: list[str] = []

    if approved_hints:
        missing_label_approval = approve_missing_label_hints_view(
            dataset_root=dataset_root,
            approved_hints=approved_hints,
            allow_overwrite=allow_overwrite_missing_label_files,
        )
        approved_label_files += missing_label_approval.approved_label_files
        applied_proposals += missing_label_approval.approved_detections
        label_paths.extend(missing_label_approval.label_paths)

    if approved_bbox_items:
        bbox_audit_approval = approve_bbox_audit_view(
            dataset_root=dataset_root,
            approved_items=approved_bbox_items,
        )
        approved_label_files += bbox_audit_approval.approved_label_files
        applied_proposals += bbox_audit_approval.applied_proposals
        label_paths.extend(bbox_audit_approval.label_paths)

    return DetectorReviewApprovalViewModel(
        approved_label_files=approved_label_files,
        applied_proposals=applied_proposals,
        label_paths=label_paths,
    )


def approve_detector_review_edited_boxes_view(
    *,
    dataset_root: Path,
    review_item: DetectorReviewItemViewModel,
    edited_boxes: list[DetectorReviewEditorBoxViewModel],
    allow_overwrite_missing_label_files: bool = False,
) -> DetectorReviewApprovalViewModel:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    label_path = safe_resolve(resolved_root, review_item.label_rel_path)
    if review_item.source_kind == "missing_label":
        if label_path.exists() and not allow_overwrite_missing_label_files:
            raise FileExistsError(f"Refusing to overwrite existing label file: {label_path}")
    elif review_item.source_kind == "bbox_audit":
        if not label_path.exists() or not label_path.is_file():
            raise FileNotFoundError(f"Label file not found for bbox audit approval: {label_path}")
    else:
        raise ValueError(f"Unsupported detector review item source: {review_item.source_kind}")

    detections = [
        HintDetection(
            class_id=box.class_id,
            class_name=box.class_name,
            confidence=1.0,
            bbox_xywh_normalized=box.bbox_xywh_normalized,
        )
        for box in edited_boxes
    ]
    write_yolo_hint_label_file(label_path, detections)

    return DetectorReviewApprovalViewModel(
        approved_label_files=1,
        applied_proposals=len(edited_boxes),
        label_paths=[str(label_path)],
    )


def convert_view(
    *,
    input_path: Path,
    output_path: Path,
    src: str,
    dst: str,
    map_path: Path | None,
    unmapped_policy: str,
    dry_run: bool,
    custom_format_id: str | None = None,
    custom_format_path: Path | None = None,
    copy_images: bool = False,
    allow_overwrite: bool = False,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    output_file_name_prefix: str | None = None,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
    flatten_output_layout: bool = False,
    drop_frames_with_class_ids: frozenset[int] | None = None,
    validation_mode: str = ValidationMode.STRICT.value,
    permissive_invalid_annotation_action: str = InvalidAnnotationAction.KEEP.value,
    out_of_frame_bbox_policy: str = DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
    out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    min_image_longest_edge_px: int = DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    max_image_longest_edge_px: int = DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    oversize_image_action: str = OversizeImageAction.IGNORE.value,
    progress_callback: ConversionProgressCallback | None = None,
) -> tuple[ConversionViewModel, ConvertResult]:
    run_id = generate_run_id("gui")
    class_map = load_mapping_file(map_path) if map_path else {}

    result = execute_conversion(
        ConvertRequest(
            run_id=run_id,
            input_path=input_path,
            output_path=output_path,
            src_format=SourceFormat(src),
            dst_format=SourceFormat(dst),
            custom_format_id=custom_format_id,
            custom_format_path=custom_format_path,
            class_map=class_map,
            drop_frames_with_class_ids=drop_frames_with_class_ids or frozenset(),
            unmapped_policy=UnmappedPolicy(unmapped_policy),
            dry_run=dry_run,
            force_infer=True,
            copy_images=copy_images,
            allow_overwrite=allow_overwrite,
            input_path_include_substring=input_path_include_substring,
            input_path_exclude_substring=input_path_exclude_substring,
            output_file_name_prefix=output_file_name_prefix,
            output_file_stem_prefix=output_file_stem_prefix,
            output_file_stem_suffix=output_file_stem_suffix,
            flatten_output_layout=flatten_output_layout,
            validation_mode=ValidationMode(validation_mode),
            permissive_invalid_annotation_action=InvalidAnnotationAction(permissive_invalid_annotation_action),
            out_of_frame_bbox_policy=OutOfFrameBBoxPolicy(out_of_frame_bbox_policy),
            out_of_frame_tolerance_px=out_of_frame_tolerance_px,
            min_image_longest_edge_px=min_image_longest_edge_px,
            max_image_longest_edge_px=max_image_longest_edge_px,
            oversize_image_action=OversizeImageAction(oversize_image_action),
        ),
        progress_callback=progress_callback,
    )

    vm = ConversionViewModel(
        run_id=run_id,
        annotations_in=result.report.summary_counts.annotations_in,
        annotations_out=result.report.summary_counts.annotations_out,
        dropped=result.report.summary_counts.dropped,
        unmapped=result.report.summary_counts.unmapped,
        contention_events=len(result.report.contention_events),
    )
    return vm, result


def build_gui_run_config(
    *,
    run_id: str,
    input_path: Path,
    output_path: Path,
    src: str,
    dst: str,
    map_path: Path | None,
    unmapped_policy: str,
    dry_run: bool,
    custom_format_id: str | None = None,
    custom_format_path: Path | None = None,
    allow_overwrite: bool = False,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    validation_mode: str = ValidationMode.STRICT.value,
    permissive_invalid_annotation_action: str = InvalidAnnotationAction.KEEP.value,
    out_of_frame_bbox_policy: str = DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
    out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    min_image_longest_edge_px: int = DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    max_image_longest_edge_px: int = DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    oversize_image_action: str = OversizeImageAction.IGNORE.value,
) -> RunConfigModel:
    return RunConfigModel(
        run_id=run_id,
        mode="convert",
        input_path=str(input_path),
        output_path=str(output_path),
        src_format=src,
        custom_format_id=custom_format_id,
        custom_format_path=str(custom_format_path) if custom_format_path else None,
        dst_format=dst,  # type: ignore[arg-type]
        mapping_file=str(map_path) if map_path else None,
        unmapped_policy=unmapped_policy,  # type: ignore[arg-type]
        dry_run=dry_run,
        allow_overwrite=allow_overwrite,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        validation_mode=validation_mode,  # type: ignore[arg-type]
        permissive_invalid_annotation_action=permissive_invalid_annotation_action,  # type: ignore[arg-type]
        out_of_frame_bbox_policy=out_of_frame_bbox_policy,  # type: ignore[arg-type]
        out_of_frame_tolerance_px=out_of_frame_tolerance_px,
        min_image_longest_edge_px=min_image_longest_edge_px,
        max_image_longest_edge_px=max_image_longest_edge_px,
        oversize_image_action=oversize_image_action,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )
