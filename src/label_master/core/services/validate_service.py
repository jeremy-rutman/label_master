from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from label_master.adapters.coco.reader import read_coco_dataset
from label_master.adapters.yolo.reader import read_yolo_dataset
from label_master.core.domain.entities import AnnotationDataset, SourceFormat, ValidationSummary
from label_master.core.domain.policies import ValidationMode, ValidationPolicy
from label_master.core.domain.value_objects import ValidationError
from label_master.core.services.infer_service import infer_format


class ValidationOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataset: AnnotationDataset
    inferred_format: SourceFormat
    summary: ValidationSummary


class _ManifestInferenceExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_candidate: Literal["coco", "yolo"]
    min_confidence: float = Field(ge=0.0, le=1.0)
    allow_ambiguous: bool = False


class _ManifestValidationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_result: Literal["pass", "fail"]
    max_invalid_annotations: int = Field(ge=0)


class _ManifestExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_format: Literal["coco", "yolo"]
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


def _load_dataset(path: Path, source_format: SourceFormat) -> AnnotationDataset:
    if source_format == SourceFormat.COCO:
        return read_coco_dataset(path)
    if source_format == SourceFormat.YOLO:
        return read_yolo_dataset(path)
    raise ValidationError(f"Unsupported source format for loading: {source_format.value}")


def validate_dataset(
    input_path: Path,
    *,
    source_format: SourceFormat = SourceFormat.AUTO,
    policy: ValidationPolicy | None = None,
) -> ValidationOutcome:
    policy = policy or ValidationPolicy(mode=ValidationMode.STRICT)

    inferred = infer_format(input_path, force=True)
    resolved_format = source_format
    if source_format == SourceFormat.AUTO:
        if inferred.predicted_format in {SourceFormat.COCO, SourceFormat.YOLO}:
            resolved_format = inferred.predicted_format
        else:
            raise ValidationError(
                "Unable to resolve source format for validation",
                context={"predicted": inferred.predicted_format.value},
            )

    dataset = _load_dataset(input_path, resolved_format)

    invalid = 0
    errors: list[str] = []
    images_by_id = {image.image_id: image for image in dataset.images}

    for annotation in dataset.annotations:
        image = images_by_id.get(annotation.image_id)
        if image is None:
            invalid += 1
            errors.append(f"Annotation {annotation.annotation_id} references unknown image")
            continue

        x, y, w, h = annotation.bbox_xywh_abs
        if w <= 0 or h <= 0:
            invalid += 1
            errors.append(f"Annotation {annotation.annotation_id} has non-positive bbox size")
            continue

        if image.checksum == "unknown_size":
            continue

        if x < 0 or y < 0 or x + w > image.width or y + h > image.height:
            invalid += 1
            errors.append(f"Annotation {annotation.annotation_id} bbox out of bounds")

    valid = invalid <= policy.max_invalid_annotations
    summary = ValidationSummary(valid=valid, invalid_annotations=invalid, errors=errors)

    if policy.mode == ValidationMode.STRICT and not valid:
        raise ValidationError(
            "Validation failed in strict mode",
            context={"invalid_annotations": str(invalid)},
        )

    return ValidationOutcome(dataset=dataset, inferred_format=resolved_format, summary=summary)


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
