from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from label_master.adapters.matlab_ground_truth.reader import read_matlab_ground_truth_dataset
from label_master.adapters.video_bbox.reader import read_video_bbox_dataset
from label_master.adapters.voc.reader import read_voc_dataset
from label_master.adapters.yolo.reader import read_yolo_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    InferencePolicy,
    InvalidAnnotationAction,
    OversizeImageAction,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.services.convert_service import (
    ConversionProgressCallback,
    ConvertRequest,
    ConvertResult,
    execute_conversion,
)
from label_master.core.services.infer_service import infer_format
from label_master.core.services.missing_label_hint_service import generate_missing_yolo_label_hints
from label_master.core.services.validate_service import validate_dataset, validate_loaded_dataset
from label_master.infra.config import load_mapping_file
from label_master.infra.filesystem import build_input_path_filter
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
    errors: list[str]


def _preview_invalid_annotation_warning(
    invalid_annotations: int,
    errors: list[str],
    *,
    correct_out_of_frame_bboxes: bool,
    out_of_frame_tolerance_px: float,
) -> str | None:
    if invalid_annotations <= 0:
        return None

    frame_overflow_errors = sum("bbox goes out of frame" in error for error in errors)
    if frame_overflow_errors == invalid_annotations:
        if not correct_out_of_frame_bboxes:
            return (
                f"Preview loaded with {invalid_annotations} invalid annotation(s): "
                "bbox goes out of frame. Enable 'Correct out-of-frame bboxes' in Step 3 to clip near-edge boxes."
            )
        return (
            f"Preview loaded with {invalid_annotations} invalid annotation(s): "
            f"bbox goes out of frame beyond the accepted {out_of_frame_tolerance_px:g}px correction tolerance."
        )
    if frame_overflow_errors:
        if not correct_out_of_frame_bboxes:
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
    correct_out_of_frame_bboxes: bool,
    out_of_frame_tolerance_px: float,
    input_path_include_substring: str | None,
    input_path_exclude_substring: str | None,
    preview_scan_limit: int,
) -> DatasetPreviewViewModel:
    resolved_format = SourceFormat(source_format)
    if resolved_format not in {
        SourceFormat.COCO,
        SourceFormat.CUSTOM,
        SourceFormat.KITWARE,
        SourceFormat.MATLAB_GROUND_TRUTH,
        SourceFormat.VOC,
        SourceFormat.VIDEO_BBOX,
        SourceFormat.YOLO,
    }:
        raise ValueError(
            "Preview source format must be coco, custom, kitware, matlab_ground_truth, voc, video_bbox, or yolo"
        )

    preview_policy = ValidationPolicy.for_mode(
        ValidationMode.PERMISSIVE,
        correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
    else:
        validation = validate_dataset(
            Path(input_path),
            source_format=resolved_format,
            policy=preview_policy,
            input_path_include_substring=input_path_include_substring,
            input_path_exclude_substring=input_path_exclude_substring,
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
        correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
    correct_out_of_frame_bboxes: bool = DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
    out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    preview_scan_limit: int = 0,
) -> DatasetPreviewViewModel:
    normalized_preview_scan_limit = max(0, int(preview_scan_limit))
    return _preview_dataset_view_cached(
        str(input_path.expanduser().resolve()),
        source_format,
        correct_out_of_frame_bboxes,
        float(out_of_frame_tolerance_px),
        input_path_include_substring,
        input_path_exclude_substring,
        normalized_preview_scan_limit,
    )


def parse_mapping_rows(rows: list[MappingRowViewModel]) -> MappingParseViewModel:
    class_map: dict[int, int | None] = {}
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

        if action not in {"map", "drop"}:
            errors.append(f"Row {index}: action must be 'map' or 'drop'")
            continue

        if action == "drop":
            class_map[source_class_id] = None
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

    return MappingParseViewModel(class_map=class_map, errors=errors)


def infer_view(input_path: Path, *, sample_limit: int = 100) -> InferenceViewModel:
    result = infer_format(input_path, policy=InferencePolicy(sample_limit=sample_limit), force=True)
    return InferenceViewModel(
        predicted_format=result.predicted_format.value,
        confidence=result.confidence,
        candidates=[(candidate.format.value, candidate.score) for candidate in result.candidates],
        warnings=[warning.message for warning in result.warnings],
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
    sample_hints = [
        {
            "image": hint.image_rel_path,
            "suggested_label": hint.suggested_label_rel_path,
            "detections": len(hint.detections),
        }
        for hint in result.hints[:50]
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
    copy_images: bool = False,
    allow_overwrite: bool = False,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    output_file_name_prefix: str | None = None,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
    flatten_output_layout: bool = False,
    validation_mode: str = ValidationMode.STRICT.value,
    permissive_invalid_annotation_action: str = InvalidAnnotationAction.KEEP.value,
    correct_out_of_frame_bboxes: bool = DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
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
            class_map=class_map,
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
            correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
    allow_overwrite: bool = False,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    validation_mode: str = ValidationMode.STRICT.value,
    permissive_invalid_annotation_action: str = InvalidAnnotationAction.KEEP.value,
    correct_out_of_frame_bboxes: bool = DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
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
        dst_format=dst,  # type: ignore[arg-type]
        mapping_file=str(map_path) if map_path else None,
        unmapped_policy=unmapped_policy,  # type: ignore[arg-type]
        dry_run=dry_run,
        allow_overwrite=allow_overwrite,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        validation_mode=validation_mode,  # type: ignore[arg-type]
        permissive_invalid_annotation_action=permissive_invalid_annotation_action,  # type: ignore[arg-type]
        correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
        out_of_frame_tolerance_px=out_of_frame_tolerance_px,
        min_image_longest_edge_px=min_image_longest_edge_px,
        max_image_longest_edge_px=max_image_longest_edge_px,
        oversize_image_action=oversize_image_action,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )
