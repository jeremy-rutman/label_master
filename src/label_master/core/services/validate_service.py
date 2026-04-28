from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from label_master.adapters.coco.reader import read_coco_dataset
from label_master.adapters.custom.reader import read_custom_dataset
from label_master.adapters.kitware.reader import read_kitware_dataset
from label_master.adapters.matlab_ground_truth.reader import read_matlab_ground_truth_dataset
from label_master.adapters.video_bbox.reader import read_video_bbox_dataset
from label_master.adapters.voc.reader import read_voc_dataset
from label_master.adapters.yolo.reader import read_yolo_dataset
from label_master.core.domain.entities import (
    AnnotationDataset,
    Severity,
    SourceFormat,
    ValidationSummary,
    WarningEvent,
)
from label_master.core.domain.policies import (
    InvalidAnnotationAction,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.core.services.infer_service import infer_format
from label_master.format_specs.registry import resolve_builtin_format_spec
from label_master.infra.filesystem import (
    InputPathFilter,
    build_input_path_filter,
    relative_path_matches_input_filter,
)
from label_master.reports.schemas import DroppedAnnotationModel

_BUILTIN_READERS = {
    "coco": read_coco_dataset,
    "kitware": read_kitware_dataset,
    "matlab_ground_truth": read_matlab_ground_truth_dataset,
    "voc": read_voc_dataset,
    "video_bbox": read_video_bbox_dataset,
    "yolo": read_yolo_dataset,
}


class ValidationOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataset: AnnotationDataset
    inferred_format: SourceFormat
    summary: ValidationSummary
    warnings: list[WarningEvent] = Field(default_factory=list)
    dropped_annotations: list[DroppedAnnotationModel] = Field(default_factory=list)


class _ManifestInferenceExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_candidate: Literal["coco", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
    min_confidence: float = Field(ge=0.0, le=1.0)
    allow_ambiguous: bool = False


class _ManifestValidationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_result: Literal["pass", "fail"]
    max_invalid_annotations: int = Field(ge=0)


class _ManifestExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_format: Literal["coco", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
    inference: _ManifestInferenceExpectation
    validation: _ManifestValidationExpectation


class _ManifestTolerance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abs: float = Field(ge=0.0)
    rel: float = Field(ge=0.0)


class _ManifestExpectedBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str
    class_id: int
    bbox: tuple[float, float, float, float]
    annotation_ref: str
    required: bool = True


class _ManifestBBoxChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_space: Literal["xywh_pixels", "normalized_cxcywh"]
    tolerance: _ManifestTolerance
    expected_boxes: list[_ManifestExpectedBox]


class _DryRunExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expect_converted_outputs_written: bool = False
    expected_exit_code: int = 0


class DryRunSampleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: int = Field(ge=1)
    sample_id: str
    description: str
    dataset_root: str
    expected: _ManifestExpected
    bbox_checks: _ManifestBBoxChecks
    dry_run_expectations: _DryRunExpectations
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DryRunVerificationResult:
    sample_id: str
    success: bool
    diagnostics: list[str]
    expected_exit_code: int


DatasetLoadProgressCallback = Callable[[int, int], None]
AnnotationValidationProgressCallback = Callable[[int, int], None]


def _load_dataset(
    path: Path,
    source_format: SourceFormat,
    *,
    load_progress_callback: DatasetLoadProgressCallback | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    if source_format == SourceFormat.CUSTOM:
        return read_custom_dataset(path, input_path_filter=input_path_filter)
    if source_format == SourceFormat.MATLAB_GROUND_TRUTH:
        return read_matlab_ground_truth_dataset(
            path,
            progress_callback=load_progress_callback,
            input_path_filter=input_path_filter,
        )
    if source_format == SourceFormat.COCO:
        return read_coco_dataset(path, input_path_filter=input_path_filter)
    if source_format == SourceFormat.KITWARE:
        return read_kitware_dataset(path, input_path_filter=input_path_filter)
    if source_format == SourceFormat.VOC:
        return read_voc_dataset(path, input_path_filter=input_path_filter)
    if source_format == SourceFormat.VIDEO_BBOX:
        return read_video_bbox_dataset(path, input_path_filter=input_path_filter)
    if source_format == SourceFormat.YOLO:
        return read_yolo_dataset(path, input_path_filter=input_path_filter)

    if resolve_builtin_format_spec(source_format.value) is not None:
        reader = _BUILTIN_READERS.get(source_format.value)
        if reader is not None:
            return reader(path)
    raise ValidationError(f"Unsupported source format for loading: {source_format.value}")


def _apply_input_path_filter_to_dataset(
    dataset: AnnotationDataset,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    if input_path_filter is None or not input_path_filter.is_active:
        return dataset

    original_image_count = len(dataset.images)
    original_annotation_count = len(dataset.annotations)
    filtered_images = [
        image
        for image in dataset.images
        if relative_path_matches_input_filter(
            image.file_name,
            input_path_filter=input_path_filter,
        )
    ]

    filtered_image_ids = {image.image_id for image in filtered_images}
    filtered_annotations = [
        annotation
        for annotation in dataset.annotations
        if annotation.image_id in filtered_image_ids
    ]
    details = dict(dataset.source_metadata.details)
    if input_path_filter.include_substring:
        details["input_path_include_substring"] = input_path_filter.include_substring
    if input_path_filter.exclude_substring:
        details["input_path_exclude_substring"] = input_path_filter.exclude_substring
    details["images_filtered_out_by_input_path"] = str(len(dataset.images) - len(filtered_images))
    details["annotations_filtered_out_by_input_path"] = str(
        len(dataset.annotations) - len(filtered_annotations)
    )

    warnings = list(dataset.warnings)
    warnings.append(
        WarningEvent(
            code="input_path_filter_applied",
            message=(
                f"Applied input path filter; dataset now contains {len(filtered_images)} images "
                f"and {len(filtered_annotations)} annotations."
                if len(filtered_images) == original_image_count
                and len(filtered_annotations) == original_annotation_count
                else (
                    f"Applied input path filter; kept {len(filtered_images)} of {original_image_count} images "
                    f"and {len(filtered_annotations)} of {original_annotation_count} annotations."
                )
            ),
            severity=Severity.INFO,
            context={
                "images_kept": str(len(filtered_images)),
                "images_total": str(original_image_count),
                "annotations_kept": str(len(filtered_annotations)),
                "annotations_total": str(original_annotation_count),
                "include_substring": input_path_filter.include_substring or "",
                "exclude_substring": input_path_filter.exclude_substring or "",
            },
        )
    )
    return dataset.model_copy(
        update={
            "images": filtered_images,
            "annotations": filtered_annotations,
            "warnings": warnings,
            "source_metadata": dataset.source_metadata.model_copy(update={"details": details}),
        }
    )


def _clip_bbox_to_image_bounds(
    bbox_xywh_abs: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    tolerance_px: float,
) -> tuple[tuple[float, float, float, float] | None, bool]:
    x, y, w, h = bbox_xywh_abs
    image_width_f = float(image_width)
    image_height_f = float(image_height)

    left_overflow = max(0.0, -x)
    top_overflow = max(0.0, -y)
    right_overflow = max(0.0, (x + w) - image_width_f)
    bottom_overflow = max(0.0, (y + h) - image_height_f)
    overflow = max(left_overflow, top_overflow, right_overflow, bottom_overflow)

    if overflow == 0.0:
        return bbox_xywh_abs, False

    if overflow > tolerance_px:
        return None, False

    clipped_x = min(max(x, 0.0), image_width_f)
    clipped_y = min(max(y, 0.0), image_height_f)
    clipped_right = min(max(x + w, 0.0), image_width_f)
    clipped_bottom = min(max(y + h, 0.0), image_height_f)
    clipped_w = clipped_right - clipped_x
    clipped_h = clipped_bottom - clipped_y

    if clipped_w <= 0 or clipped_h <= 0:
        return None, False

    return (clipped_x, clipped_y, clipped_w, clipped_h), True


def _format_scalar(value: float) -> str:
    return f"{value:.2f}"


def _format_bbox_xywh_abs(bbox_xywh_abs: tuple[float, float, float, float]) -> str:
    x, y, w, h = bbox_xywh_abs
    return f"({_format_scalar(x)}, {_format_scalar(y)}, {_format_scalar(w)}, {_format_scalar(h)})"


def _format_frame_bounds(*, image_width: int, image_height: int) -> str:
    return f"width={image_width}, height={image_height}"


def _format_bbox_overflow_px(
    bbox_xywh_abs: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> str:
    x, y, w, h = bbox_xywh_abs
    overflow_components = {
        "left": max(0.0, -x),
        "top": max(0.0, -y),
        "right": max(0.0, (x + w) - float(image_width)),
        "bottom": max(0.0, (y + h) - float(image_height)),
    }
    return ", ".join(
        f"{side}={_format_scalar(value)}"
        for side, value in overflow_components.items()
        if value > 0.0
    )


def _build_issue_row(
    *,
    issue_kind: str,
    issue: str,
    annotation_id: str,
    image_id: str,
    bbox_xywh_abs: tuple[float, float, float, float],
    image_file: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    include_overflow: bool = False,
) -> dict[str, str]:
    row = {
        "kind": issue_kind,
        "issue": issue,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "bbox_xywh_abs": _format_bbox_xywh_abs(bbox_xywh_abs),
    }
    if image_file:
        row["image_file"] = image_file
    if image_width is not None and image_height is not None:
        row["frame_bounds"] = _format_frame_bounds(image_width=image_width, image_height=image_height)
        if include_overflow:
            overflow_px = _format_bbox_overflow_px(
                bbox_xywh_abs,
                image_width=image_width,
                image_height=image_height,
            )
            if overflow_px:
                row["overflow_px"] = overflow_px
    return row


def _issue_message(row: dict[str, str]) -> str:
    base = f"Annotation {row['annotation_id']} {row['issue']}"
    detail_parts: list[str] = []
    for key in ("image_id", "image_file", "bbox_xywh_abs", "frame_bounds", "overflow_px"):
        value = row.get(key)
        if not value:
            continue
        detail_parts.append(f"{key}={value}")
    if not detail_parts:
        return base
    return f"{base}: " + ", ".join(detail_parts)


def _dropped_annotation_context(issue_row: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in issue_row.items()
        if key not in {"kind", "annotation_id", "image_id", "image_file", "issue", "bbox_xywh_abs"} and value
    }


def _build_dropped_annotation(
    *,
    annotation: Any,
    categories: dict[int, Any],
    stage: Literal["validation", "remap", "size_gate"],
    reason_code: str,
    reason: str,
    image_file: str | None = None,
    context: dict[str, str] | None = None,
) -> DroppedAnnotationModel:
    class_id = int(getattr(annotation, "class_id"))
    category = categories.get(class_id)
    class_name = getattr(category, "name", None)
    bbox_xywh_abs = getattr(annotation, "bbox_xywh_abs")
    return DroppedAnnotationModel(
        annotation_id=str(getattr(annotation, "annotation_id")),
        image_id=str(getattr(annotation, "image_id")),
        image_file=image_file,
        class_id=class_id,
        class_name=class_name if isinstance(class_name, str) and class_name else None,
        bbox_xywh_abs=(
            float(bbox_xywh_abs[0]),
            float(bbox_xywh_abs[1]),
            float(bbox_xywh_abs[2]),
            float(bbox_xywh_abs[3]),
        ),
        stage=stage,
        reason_code=reason_code,
        reason=reason,
        context=context or {},
    )


def _build_dropped_source_file(
    *,
    source_file: str,
    reason_code: str,
    reason: str,
    context: dict[str, str] | None = None,
) -> DroppedAnnotationModel:
    return DroppedAnnotationModel(
        source_file=source_file,
        stage="load",
        reason_code=reason_code,
        reason=reason,
        context=context or {},
    )


def _dropped_annotations_from_loader_warnings(warnings: list[WarningEvent]) -> list[DroppedAnnotationModel]:
    dropped_annotations: list[DroppedAnnotationModel] = []

    for warning in warnings:
        skipped_files_json = warning.context.get("skipped_files_json")
        if skipped_files_json:
            try:
                payload = json.loads(skipped_files_json)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    source_file = str(
                        item.get("source_file") or item.get("xml_file") or ""
                    ).strip()
                    reason = str(item.get("reason") or warning.message).strip()
                    if not source_file or not reason:
                        continue
                    dropped_annotations.append(
                        _build_dropped_source_file(
                            source_file=source_file,
                            reason_code=warning.code,
                            reason=reason,
                            context={"source_file": source_file, "reason": reason},
                        )
                    )
                continue

        source_file = str(
            warning.context.get("source_file") or warning.context.get("xml_file") or ""
        ).strip()
        reason = str(warning.context.get("reason") or "").strip()
        if source_file and reason:
            dropped_annotations.append(
                _build_dropped_source_file(
                    source_file=source_file,
                    reason_code=warning.code,
                    reason=reason,
                    context={"source_file": source_file, "reason": reason},
                )
            )

    return dropped_annotations


def validate_loaded_dataset(
    dataset: AnnotationDataset,
    *,
    source_format: SourceFormat,
    policy: ValidationPolicy | None = None,
    annotation_progress_callback: AnnotationValidationProgressCallback | None = None,
) -> ValidationOutcome:
    policy = policy or ValidationPolicy(mode=ValidationMode.STRICT)

    invalid = 0
    clipped = 0
    dropped_invalid = 0
    errors: list[str] = []
    issue_rows_sample: list[dict[str, str]] = []
    warnings: list[WarningEvent] = list(dataset.warnings)
    dropped_annotations: list[DroppedAnnotationModel] = _dropped_annotations_from_loader_warnings(warnings)
    images_by_id = {image.image_id: image for image in dataset.images}
    corrected_annotations = []
    total_annotations = len(dataset.annotations)
    progress_update_every = max(total_annotations // 100, 1) if total_annotations else 1

    for annotation_index, annotation in enumerate(dataset.annotations, start=1):
        try:
            image = images_by_id.get(annotation.image_id)
            if image is None:
                invalid += 1
                issue_row = _build_issue_row(
                    issue_kind="First issue" if not issue_rows_sample else "Sample issue",
                    issue="references unknown image",
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    bbox_xywh_abs=annotation.bbox_xywh_abs,
                )
                errors.append(_issue_message(issue_row))
                if len(issue_rows_sample) < 5:
                    issue_rows_sample.append(issue_row)
                if (
                    policy.mode == ValidationMode.PERMISSIVE
                    and policy.invalid_annotation_action == InvalidAnnotationAction.DROP
                ):
                    dropped_invalid += 1
                    dropped_annotations.append(
                        _build_dropped_annotation(
                            annotation=annotation,
                            categories=dataset.categories,
                            stage="validation",
                            reason_code="missing_image_reference",
                            reason="references unknown image",
                            context=_dropped_annotation_context(issue_row),
                        )
                    )
                    continue
                corrected_annotations.append(annotation)
                continue

            x, y, w, h = annotation.bbox_xywh_abs
            if w <= 0 or h <= 0:
                invalid += 1
                issue_row = _build_issue_row(
                    issue_kind="First issue" if not issue_rows_sample else "Sample issue",
                    issue="has non-positive bbox size",
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    bbox_xywh_abs=annotation.bbox_xywh_abs,
                    image_file=image.file_name,
                    image_width=image.width,
                    image_height=image.height,
                )
                errors.append(_issue_message(issue_row))
                if len(issue_rows_sample) < 5:
                    issue_rows_sample.append(issue_row)
                if (
                    policy.mode == ValidationMode.PERMISSIVE
                    and policy.invalid_annotation_action == InvalidAnnotationAction.DROP
                ):
                    dropped_invalid += 1
                    dropped_annotations.append(
                        _build_dropped_annotation(
                            annotation=annotation,
                            categories=dataset.categories,
                            stage="validation",
                            reason_code="non_positive_bbox_size",
                            reason="has non-positive bbox size",
                            image_file=image.file_name,
                            context=_dropped_annotation_context(issue_row),
                        )
                    )
                    continue
                corrected_annotations.append(annotation)
                continue

            if image.checksum == "unknown_size":
                corrected_annotations.append(annotation)
                continue

            clipped_bbox, did_clip = _clip_bbox_to_image_bounds(
                annotation.bbox_xywh_abs,
                image_width=image.width,
                image_height=image.height,
                tolerance_px=policy.out_of_frame_tolerance_px,
            )
            if clipped_bbox is None:
                invalid += 1
                issue_text = "bbox goes out of frame"
                if policy.correct_out_of_frame_bboxes:
                    issue_text = (
                        "bbox goes out of frame beyond the accepted "
                        f"{policy.out_of_frame_tolerance_px:g}px correction tolerance"
                    )
                issue_row = _build_issue_row(
                    issue_kind="First issue" if not issue_rows_sample else "Sample issue",
                    issue=issue_text,
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    bbox_xywh_abs=annotation.bbox_xywh_abs,
                    image_file=image.file_name,
                    image_width=image.width,
                    image_height=image.height,
                    include_overflow=True,
                )
                errors.append(_issue_message(issue_row))
                if len(issue_rows_sample) < 5:
                    issue_rows_sample.append(issue_row)
                if (
                    policy.mode == ValidationMode.PERMISSIVE
                    and policy.invalid_annotation_action == InvalidAnnotationAction.DROP
                ):
                    dropped_invalid += 1
                    dropped_annotations.append(
                        _build_dropped_annotation(
                            annotation=annotation,
                            categories=dataset.categories,
                            stage="validation",
                            reason_code="bbox_out_of_frame",
                            reason=issue_text,
                            image_file=image.file_name,
                            context=_dropped_annotation_context(issue_row),
                        )
                    )
                    continue
                corrected_annotations.append(annotation)
                continue

            if did_clip:
                if policy.correct_out_of_frame_bboxes:
                    clipped += 1
                    corrected_annotations.append(annotation.model_copy(update={"bbox_xywh_abs": clipped_bbox}))
                    continue
                invalid += 1
                issue_row = _build_issue_row(
                    issue_kind="First issue" if not issue_rows_sample else "Sample issue",
                    issue="bbox goes out of frame",
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    bbox_xywh_abs=annotation.bbox_xywh_abs,
                    image_file=image.file_name,
                    image_width=image.width,
                    image_height=image.height,
                    include_overflow=True,
                )
                errors.append(_issue_message(issue_row))
                if len(issue_rows_sample) < 5:
                    issue_rows_sample.append(issue_row)
                if (
                    policy.mode == ValidationMode.PERMISSIVE
                    and policy.invalid_annotation_action == InvalidAnnotationAction.DROP
                ):
                    dropped_invalid += 1
                    dropped_annotations.append(
                        _build_dropped_annotation(
                            annotation=annotation,
                            categories=dataset.categories,
                            stage="validation",
                            reason_code="bbox_out_of_frame",
                            reason="bbox goes out of frame",
                            image_file=image.file_name,
                            context=_dropped_annotation_context(issue_row),
                        )
                    )
                    continue
                corrected_annotations.append(annotation)
                continue

            corrected_annotations.append(annotation)
        finally:
            if annotation_progress_callback is not None and (
                annotation_index % progress_update_every == 0 or annotation_index == total_annotations
            ):
                annotation_progress_callback(annotation_index, total_annotations)

    if clipped:
        warnings.append(
            WarningEvent(
                code="validation_bbox_clipped_to_frame",
                message=(
                    f"Auto-corrected {clipped} annotation(s) whose bbox went slightly out of frame "
                    f"by clipping them to the image bounds (tolerance: <= "
                    f"{policy.out_of_frame_tolerance_px:g}px)."
                ),
                severity=Severity.WARNING,
                context={
                    "clipped_annotations": str(clipped),
                    "clip_tolerance_px": f"{policy.out_of_frame_tolerance_px:g}",
                },
            )
        )

    if dropped_invalid:
        warnings.append(
            WarningEvent(
                code="validation_invalid_annotations_dropped",
                message=(
                    f"Dropped {dropped_invalid} invalid annotation(s) during permissive validation."
                ),
                severity=Severity.WARNING,
                context={
                    "dropped_invalid_annotations": str(dropped_invalid),
                    "invalid_annotation_action": policy.invalid_annotation_action.value,
                },
            )
        )

    valid = invalid <= policy.max_invalid_annotations
    summary = ValidationSummary(valid=valid, invalid_annotations=invalid, errors=errors)
    dataset = dataset.model_copy(update={"annotations": corrected_annotations})

    if policy.mode == ValidationMode.STRICT and not valid:
        context = {"invalid_annotations": str(invalid)}
        if errors:
            context["first_error"] = errors[0]
            context["sample_errors"] = "\n".join(errors[:5])
        if issue_rows_sample:
            context["issue_rows_json"] = json.dumps(issue_rows_sample)
        raise ValidationError(
            f"Validation failed in strict mode: {invalid} invalid annotation(s)",
            context=context,
        )

    return ValidationOutcome(
        dataset=dataset,
        inferred_format=source_format,
        summary=summary,
        warnings=warnings,
        dropped_annotations=dropped_annotations,
    )


def validate_dataset(
    input_path: Path,
    *,
    source_format: SourceFormat = SourceFormat.AUTO,
    policy: ValidationPolicy | None = None,
    load_progress_callback: DatasetLoadProgressCallback | None = None,
    annotation_progress_callback: AnnotationValidationProgressCallback | None = None,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> ValidationOutcome:
    policy = policy or ValidationPolicy(mode=ValidationMode.STRICT)
    input_path_filter = build_input_path_filter(
        include_substring=input_path_include_substring,
        exclude_substring=input_path_exclude_substring,
    )

    resolved_format = source_format
    if source_format == SourceFormat.AUTO:
        inferred = infer_format(input_path, force=True)
        if inferred.predicted_format in {
            SourceFormat.COCO,
            SourceFormat.CUSTOM,
            SourceFormat.KITWARE,
            SourceFormat.MATLAB_GROUND_TRUTH,
            SourceFormat.VOC,
            SourceFormat.VIDEO_BBOX,
            SourceFormat.YOLO,
        }:
            resolved_format = inferred.predicted_format
        else:
            raise ValidationError(
                "Unable to resolve source format for validation",
                context={"predicted": inferred.predicted_format.value},
            )

    dataset = _load_dataset(
        input_path,
        resolved_format,
        load_progress_callback=load_progress_callback,
        input_path_filter=input_path_filter,
    )
    dataset = _apply_input_path_filter_to_dataset(
        dataset,
        input_path_filter=input_path_filter,
    )
    return validate_loaded_dataset(
        dataset,
        source_format=resolved_format,
        policy=policy,
        annotation_progress_callback=annotation_progress_callback,
    )


def load_dry_run_manifest(manifest_path: Path) -> DryRunSampleManifest:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValidationError(f"Dry-run manifest must be a YAML mapping: {manifest_path}")
    return DryRunSampleManifest.model_validate(payload)


def _resolve_manifest_dataset_root(manifest_path: Path, dataset_root: str) -> Path:
    root = Path(dataset_root)
    if root.is_absolute():
        return root
    return (manifest_path.parent / root).resolve()


def _match_annotation(
    dataset: AnnotationDataset,
    expectation: _ManifestExpectedBox,
) -> tuple[float, float, float, float] | None:
    for annotation in dataset.annotations:
        image = next((img for img in dataset.images if img.image_id == annotation.image_id), None)
        if image is None:
            continue

        annotation_ref_match = annotation.annotation_id == expectation.annotation_ref or (
            annotation.annotation_id.endswith(expectation.annotation_ref)
            or expectation.annotation_ref.endswith(annotation.annotation_id)
        )
        image_match = image.file_name == expectation.image or image.file_name.endswith(expectation.image)

        if annotation_ref_match and annotation.class_id == expectation.class_id:
            return annotation.bbox_xywh_abs

        if image_match and annotation.class_id == expectation.class_id:
            return annotation.bbox_xywh_abs

    return None


def _within_tolerance(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
    tol_abs: float,
    tol_rel: float,
) -> bool:
    for a, e in zip(actual, expected, strict=True):
        delta = abs(a - e)
        allowed = max(tol_abs, abs(e) * tol_rel)
        if delta > allowed:
            return False
    return True


def _to_normalized(
    dataset: AnnotationDataset,
    bbox: tuple[float, float, float, float],
    image_name: str,
) -> tuple[float, float, float, float]:
    image = next(
        (img for img in dataset.images if img.file_name == image_name or img.file_name.endswith(image_name)),
        None,
    )
    if image is None:
        return bbox

    x, y, w, h = bbox
    cx = (x + w / 2.0) / image.width
    cy = (y + h / 2.0) / image.height
    return (cx, cy, w / image.width, h / image.height)


def verify_dry_run_manifest(manifest_path: Path) -> DryRunVerificationResult:
    manifest = load_dry_run_manifest(manifest_path)
    diagnostics: list[str] = []

    dataset_root = _resolve_manifest_dataset_root(manifest_path, manifest.dataset_root)

    inference = infer_format(dataset_root, force=True)
    top_candidate = inference.candidates[0].format.value
    if top_candidate != manifest.expected.inference.top_candidate:
        diagnostics.append(
            f"Top candidate mismatch: expected {manifest.expected.inference.top_candidate}, got {top_candidate}"
        )

    if inference.confidence < manifest.expected.inference.min_confidence:
        diagnostics.append(
            "Inference confidence below manifest minimum: "
            f"{inference.confidence} < {manifest.expected.inference.min_confidence}"
        )

    if (
        not manifest.expected.inference.allow_ambiguous
        and inference.predicted_format == SourceFormat.AMBIGUOUS
    ):
        diagnostics.append("Inference result is ambiguous but manifest disallows ambiguity")

    source_format = SourceFormat(manifest.expected.source_format)
    mode = ValidationMode.STRICT if manifest.expected.validation.expected_result == "pass" else ValidationMode.PERMISSIVE
    policy = ValidationPolicy(mode=mode, max_invalid_annotations=manifest.expected.validation.max_invalid_annotations)

    try:
        validation = validate_dataset(dataset_root, source_format=source_format, policy=policy)
        validation_passed = validation.summary.valid
    except ValidationError as exc:
        diagnostics.append(str(exc))
        return DryRunVerificationResult(
            sample_id=manifest.sample_id,
            success=False,
            diagnostics=diagnostics,
            expected_exit_code=manifest.dry_run_expectations.expected_exit_code,
        )

    expected_pass = manifest.expected.validation.expected_result == "pass"
    if validation_passed != expected_pass:
        diagnostics.append(
            f"Validation expectation mismatch: expected {manifest.expected.validation.expected_result}, got "
            f"{'pass' if validation_passed else 'fail'}"
        )

    if validation.summary.invalid_annotations > manifest.expected.validation.max_invalid_annotations:
        diagnostics.append(
            "Invalid annotation count exceeds manifest limit: "
            f"{validation.summary.invalid_annotations} > {manifest.expected.validation.max_invalid_annotations}"
        )

    for expected_box in manifest.bbox_checks.expected_boxes:
        actual = _match_annotation(validation.dataset, expected_box)
        if actual is None:
            if expected_box.required:
                diagnostics.append(f"Expected box not found: {expected_box.annotation_ref}")
            continue

        actual_comparable = actual
        if manifest.bbox_checks.coordinate_space == "normalized_cxcywh":
            actual_comparable = _to_normalized(validation.dataset, actual, expected_box.image)

        if not _within_tolerance(
            actual_comparable,
            expected_box.bbox,
            manifest.bbox_checks.tolerance.abs,
            manifest.bbox_checks.tolerance.rel,
        ):
            diagnostics.append(
                "BBox mismatch for "
                f"{expected_box.annotation_ref}: expected {expected_box.bbox}, got {actual_comparable}"
            )

    if manifest.dry_run_expectations.expect_converted_outputs_written:
        diagnostics.append("Manifest requires converted outputs in dry-run, which is unsupported")

    return DryRunVerificationResult(
        sample_id=manifest.sample_id,
        success=not diagnostics,
        diagnostics=diagnostics,
        expected_exit_code=manifest.dry_run_expectations.expected_exit_code,
    )


def load_dry_run_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    manifest = load_dry_run_manifest(manifest_path)
    return manifest.model_dump(mode="python")
