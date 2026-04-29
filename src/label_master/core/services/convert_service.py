from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from typing import Callable, Literal

from PIL import Image

from label_master.adapters.coco.writer import write_coco_dataset
from label_master.adapters.video_bbox.reader import materialize_video_bbox_frames
from label_master.adapters.yolo.writer import (
    image_output_rel_path_for_image,
    label_output_rel_path_for_image,
    write_yolo_dataset,
)
from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    ImageRecord,
    Severity,
    SourceFormat,
    WarningEvent,
)
from label_master.core.domain.policies import (
    DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    InvalidAnnotationAction,
    OversizeImageAction,
    RemapPolicy,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.domain.value_objects import ConversionError
from label_master.core.services.infer_service import infer_format
from label_master.core.services.remap_service import RemapResult, apply_class_remap
from label_master.core.services.validate_service import (
    DryRunVerificationResult,
    ValidationOutcome,
    validate_dataset,
    verify_dry_run_manifest,
)
from label_master.infra.filesystem import ensure_directory, safe_resolve
from label_master.infra.locking import OutputPathLockManager
from label_master.reports.schemas import (
    ContentionEventModel,
    DroppedAnnotationModel,
    RunReportModel,
    SummaryCountsModel,
    WarningEventModel,
)


@dataclass(frozen=True)
class ConvertRequest:
    run_id: str
    input_path: Path
    output_path: Path | None
    src_format: SourceFormat
    dst_format: SourceFormat | None
    class_map: dict[int, int | None] = field(default_factory=dict)
    unmapped_policy: UnmappedPolicy = UnmappedPolicy.ERROR
    dry_run: bool = False
    force_infer: bool = False
    copy_images: bool = False
    allow_overwrite: bool = False
    input_path_include_substring: str | None = None
    input_path_exclude_substring: str | None = None
    output_file_name_prefix: str | None = None
    output_file_stem_prefix: str | None = None
    output_file_stem_suffix: str | None = None
    flatten_output_layout: bool = False
    validation_mode: ValidationMode = ValidationMode.STRICT
    permissive_invalid_annotation_action: InvalidAnnotationAction = InvalidAnnotationAction.KEEP
    correct_out_of_frame_bboxes: bool = DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES
    out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX
    min_image_longest_edge_px: int = DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX
    max_image_longest_edge_px: int = DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX
    oversize_image_action: OversizeImageAction = OversizeImageAction.IGNORE


@dataclass(frozen=True)
class ConvertResult:
    report: RunReportModel
    output_artifacts: list[Path]
    validation: ValidationOutcome
    remap: RemapResult | None
    output_dataset: AnnotationDataset
    dropped_annotations: list[DroppedAnnotationModel]


RunSrcFormatLiteral = Literal["auto", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
RunDstFormatLiteral = Literal["coco", "yolo"]
ConversionProgressCallback = Callable[[str, int], None]
try:
    _IMAGE_RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - compatibility with older Pillow
    _IMAGE_RESAMPLING_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def _emit_progress(
    progress_callback: ConversionProgressCallback | None,
    *,
    message: str,
    percent: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, min(max(percent, 0), 100))


def _interpolate_progress(
    *,
    start_percent: int,
    end_percent: int,
    completed_units: int,
    total_units: int,
) -> int:
    if total_units <= 0:
        return end_percent
    span = max(end_percent - start_percent, 0)
    return start_percent + int((span * min(max(completed_units, 0), total_units)) / total_units)


def _format_src_for_report(source_format: SourceFormat | None) -> RunSrcFormatLiteral | None:
    if source_format is None:
        return None
    return source_format.value  # type: ignore[return-value]


def _format_dst_for_report(source_format: SourceFormat | None) -> RunDstFormatLiteral | None:
    if source_format == SourceFormat.COCO:
        return "coco"
    if source_format == SourceFormat.YOLO:
        return "yolo"
    return None


def _build_warning_models(warnings: list[WarningEvent]) -> list[WarningEventModel]:
    return [
        WarningEventModel(
            code=item.code,
            message=item.message,
            severity=item.severity.value,
            context=item.context,
        )
        for item in warnings
    ]


def sanitize_output_file_name_prefix(raw_value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip()).strip("._-")
    return sanitized or "dataset"


def sanitize_output_file_stem_affix(raw_value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip())


def derive_output_filename_prefix(input_path: Path) -> str:
    expanded = input_path.expanduser()
    candidate = expanded.name or expanded.parent.name or "dataset"
    return sanitize_output_file_name_prefix(candidate)


def _file_name_with_stem_affixes(
    file_name: str,
    *,
    stem_prefix: str | None,
    stem_suffix: str | None,
) -> str:
    if not stem_prefix and not stem_suffix:
        return file_name

    file_path = Path(file_name)
    stem = file_path.stem or file_path.name
    affixed_name = f"{stem_prefix or ''}{stem}{stem_suffix or ''}{file_path.suffix}"
    return file_path.with_name(affixed_name).as_posix()


def _artifact_file_name(
    base_name: str,
    prefix: str | None,
    *,
    stem_prefix: str | None = None,
    stem_suffix: str | None = None,
) -> str:
    if not prefix:
        return _file_name_with_stem_affixes(
            base_name,
            stem_prefix=stem_prefix,
            stem_suffix=stem_suffix,
        )
    return _file_name_with_stem_affixes(
        f"{prefix}_{base_name}",
        stem_prefix=stem_prefix,
        stem_suffix=stem_suffix,
    )


def _prefixed_image_file_name(
    image_file_name: str,
    image_id: str,
    prefix: str,
) -> str:
    image_rel = Path(image_file_name)
    if image_rel.is_absolute():
        image_rel = Path(image_rel.name)

    parts = list(image_rel.parts)
    if len(parts) >= 3 and parts[0].lower() == "images" and image_rel.name.startswith("frame_"):
        parts[1] = f"{prefix}_{parts[1]}"
        return Path(*parts).as_posix()

    base_name = image_rel.name or image_id
    prefixed_name = f"{prefix}_{base_name}"
    parent = image_rel.parent
    if parent == Path("."):
        return prefixed_name
    return (parent / prefixed_name).as_posix()


def _dataset_with_prefixed_output_file_names(
    dataset: AnnotationDataset,
    prefix: str,
) -> AnnotationDataset:
    updated_images = [
        image.model_copy(
            update={
                "file_name": _prefixed_image_file_name(
                    image.file_name,
                    image.image_id,
                    prefix,
                )
            }
        )
        for image in dataset.images
    ]
    return dataset.model_copy(update={"images": updated_images})


def _image_file_name_with_stem_affixes(
    image_file_name: str,
    image_id: str,
    *,
    stem_prefix: str | None,
    stem_suffix: str | None,
) -> str:
    image_rel = Path(image_file_name)
    if image_rel.is_absolute():
        image_rel = Path(image_rel.name)
    if image_rel == Path(".") or not str(image_rel).strip():
        image_rel = Path(image_id)
    return _file_name_with_stem_affixes(
        image_rel.as_posix(),
        stem_prefix=stem_prefix,
        stem_suffix=stem_suffix,
    )


def _dataset_with_output_file_stem_affixes(
    dataset: AnnotationDataset,
    *,
    stem_prefix: str | None,
    stem_suffix: str | None,
) -> AnnotationDataset:
    if not stem_prefix and not stem_suffix:
        return dataset

    updated_images = [
        image.model_copy(
            update={
                "file_name": _image_file_name_with_stem_affixes(
                    image.file_name,
                    image.image_id,
                    stem_prefix=stem_prefix,
                    stem_suffix=stem_suffix,
                )
            }
        )
        for image in dataset.images
    ]
    return dataset.model_copy(update={"images": updated_images})


def _scaled_dimension(value: int, scale: float) -> int:
    scaled = max(1, int(round(value * scale)))
    return scaled


def _resized_dimensions_for_max_edge(
    width: int,
    height: int,
    max_longest_edge_px: int,
) -> tuple[int, int]:
    if max_longest_edge_px <= 0:
        return width, height
    longest_edge = max(width, height)
    if longest_edge <= max_longest_edge_px:
        return width, height

    scale = max_longest_edge_px / float(longest_edge)
    resized_width = _scaled_dimension(width, scale)
    resized_height = _scaled_dimension(height, scale)
    if max(resized_width, resized_height) > max_longest_edge_px:
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
    return resized_width, resized_height


def _annotation_with_scaled_bbox(
    annotation: AnnotationRecord,
    *,
    scale_x: float,
    scale_y: float,
) -> AnnotationRecord:
    x, y, w, h = annotation.bbox_xywh_abs
    return annotation.model_copy(
        update={
            "bbox_xywh_abs": (
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y,
            )
        }
    )


def _build_dropped_annotation(
    *,
    annotation: AnnotationRecord,
    dataset: AnnotationDataset,
    stage: Literal["validation", "remap", "size_gate"],
    reason_code: str,
    reason: str,
    image_file: str | None = None,
    context: dict[str, str] | None = None,
) -> DroppedAnnotationModel:
    category = dataset.categories.get(annotation.class_id)
    return DroppedAnnotationModel(
        annotation_id=annotation.annotation_id,
        image_id=annotation.image_id,
        image_file=image_file,
        class_id=annotation.class_id,
        class_name=category.name if category else None,
        bbox_xywh_abs=annotation.bbox_xywh_abs,
        stage=stage,
        reason_code=reason_code,
        reason=reason,
        context=context or {},
    )


def _apply_image_size_gate(
    dataset: AnnotationDataset,
    *,
    min_image_longest_edge_px: int,
    max_image_longest_edge_px: int,
    oversize_image_action: OversizeImageAction,
) -> tuple[AnnotationDataset, list[WarningEvent], list[DroppedAnnotationModel]]:
    if min_image_longest_edge_px <= 0 and max_image_longest_edge_px <= 0:
        return dataset, [], []

    annotations_by_image_id: dict[str, list[AnnotationRecord]] = {}
    for annotation in dataset.annotations:
        annotations_by_image_id.setdefault(annotation.image_id, []).append(annotation)

    transformed_images: list[ImageRecord] = []
    transformed_annotations: list[AnnotationRecord] = []
    warnings: list[WarningEvent] = []
    dropped_annotations: list[DroppedAnnotationModel] = []

    dropped_small_images = 0
    dropped_small_annotations = 0
    dropped_large_images = 0
    dropped_large_annotations = 0
    downscaled_images = 0
    unknown_dimension_images = 0

    for image in dataset.images:
        image_annotations = annotations_by_image_id.get(image.image_id, [])
        if image.checksum == "unknown_size":
            unknown_dimension_images += 1
            transformed_images.append(image)
            transformed_annotations.extend(image_annotations)
            continue

        longest_edge = max(image.width, image.height)
        if min_image_longest_edge_px > 0 and longest_edge < min_image_longest_edge_px:
            dropped_small_images += 1
            dropped_small_annotations += len(image_annotations)
            dropped_annotations.extend(
                _build_dropped_annotation(
                    annotation=annotation,
                    dataset=dataset,
                    stage="size_gate",
                    reason_code="image_below_min_longest_edge",
                    reason=(
                        f"Image dropped because its longest edge ({longest_edge}px) was below "
                        f"the minimum threshold ({min_image_longest_edge_px}px)."
                    ),
                    image_file=image.file_name,
                    context={
                        "image_width": str(image.width),
                        "image_height": str(image.height),
                        "image_longest_edge_px": str(longest_edge),
                        "min_image_longest_edge_px": str(min_image_longest_edge_px),
                    },
                )
                for annotation in image_annotations
            )
            continue

        if max_image_longest_edge_px > 0 and longest_edge > max_image_longest_edge_px:
            if oversize_image_action == OversizeImageAction.IGNORE:
                dropped_large_images += 1
                dropped_large_annotations += len(image_annotations)
                dropped_annotations.extend(
                    _build_dropped_annotation(
                        annotation=annotation,
                        dataset=dataset,
                        stage="size_gate",
                        reason_code="image_above_max_longest_edge",
                        reason=(
                            f"Image dropped because its longest edge ({longest_edge}px) exceeded "
                            f"the maximum threshold ({max_image_longest_edge_px}px)."
                        ),
                        image_file=image.file_name,
                        context={
                            "image_width": str(image.width),
                            "image_height": str(image.height),
                            "image_longest_edge_px": str(longest_edge),
                            "max_image_longest_edge_px": str(max_image_longest_edge_px),
                        },
                    )
                    for annotation in image_annotations
                )
                continue

            resized_width, resized_height = _resized_dimensions_for_max_edge(
                image.width,
                image.height,
                max_image_longest_edge_px,
            )
            if (resized_width, resized_height) != (image.width, image.height):
                scale_x = resized_width / float(image.width)
                scale_y = resized_height / float(image.height)
                transformed_images.append(
                    image.model_copy(update={"width": resized_width, "height": resized_height})
                )
                transformed_annotations.extend(
                    _annotation_with_scaled_bbox(annotation, scale_x=scale_x, scale_y=scale_y)
                    for annotation in image_annotations
                )
                downscaled_images += 1
                continue

        transformed_images.append(image)
        transformed_annotations.extend(image_annotations)

    if dropped_small_images:
        warnings.append(
            WarningEvent(
                code="size_gate_small_images_dropped",
                message=(
                    f"Dropped {dropped_small_images} image(s) and {dropped_small_annotations} annotation(s) "
                    f"whose longest edge was smaller than {min_image_longest_edge_px}px."
                ),
                severity=Severity.WARNING,
                context={
                    "images": str(dropped_small_images),
                    "annotations": str(dropped_small_annotations),
                    "min_image_longest_edge_px": str(min_image_longest_edge_px),
                },
            )
        )

    if dropped_large_images:
        warnings.append(
            WarningEvent(
                code="size_gate_large_images_dropped",
                message=(
                    f"Dropped {dropped_large_images} image(s) and {dropped_large_annotations} annotation(s) "
                    f"whose longest edge exceeded {max_image_longest_edge_px}px."
                ),
                severity=Severity.WARNING,
                context={
                    "images": str(dropped_large_images),
                    "annotations": str(dropped_large_annotations),
                    "max_image_longest_edge_px": str(max_image_longest_edge_px),
                },
            )
        )

    if downscaled_images:
        warnings.append(
            WarningEvent(
                code="size_gate_large_images_downscaled",
                message=(
                    f"Downscaled {downscaled_images} image(s) whose longest edge exceeded "
                    f"{max_image_longest_edge_px}px and updated bbox coordinates to match."
                ),
                severity=Severity.WARNING,
                context={
                    "images": str(downscaled_images),
                    "max_image_longest_edge_px": str(max_image_longest_edge_px),
                },
            )
        )

    if unknown_dimension_images:
        warnings.append(
            WarningEvent(
                code="size_gate_unknown_dimensions_skipped",
                message=(
                    f"Skipped size-gate checks for {unknown_dimension_images} image(s) whose dimensions were unknown."
                ),
                severity=Severity.WARNING,
                context={"images": str(unknown_dimension_images)},
            )
        )

    return (
        dataset.model_copy(update={"images": transformed_images, "annotations": transformed_annotations}),
        warnings,
        dropped_annotations,
    )


def _find_duplicate_output_targets(
    targets_by_output: dict[str, list[str]],
) -> list[tuple[str, list[str]]]:
    duplicates: list[tuple[str, list[str]]] = []
    for output_path, sources in sorted(targets_by_output.items()):
        unique_sources = list(dict.fromkeys(sources))
        if len(unique_sources) > 1:
            duplicates.append((output_path, unique_sources))
    return duplicates


@dataclass(frozen=True)
class PlannedOutputTarget:
    kind: str
    rel_path: str
    source_descriptor: str


@dataclass(frozen=True)
class ExistingOutputConflict:
    kind: str
    rel_path: str
    source_descriptor: str
    can_overwrite: bool


def _collect_planned_output_targets(
    dataset: AnnotationDataset,
    *,
    destination_format: SourceFormat,
    copy_images: bool,
    flatten_output_layout: bool,
    output_file_name_prefix: str | None,
    output_file_stem_prefix: str | None,
    output_file_stem_suffix: str | None,
) -> list[PlannedOutputTarget]:
    targets: list[PlannedOutputTarget] = []

    if destination_format == SourceFormat.YOLO:
        targets.append(
            PlannedOutputTarget(
                kind="artifact",
                rel_path=_artifact_file_name(
                    "classes.txt",
                    output_file_name_prefix,
                    stem_prefix=output_file_stem_prefix,
                    stem_suffix=output_file_stem_suffix,
                ),
                source_descriptor="classes.txt",
            )
        )
    elif destination_format == SourceFormat.COCO:
        targets.append(
            PlannedOutputTarget(
                kind="artifact",
                rel_path=_artifact_file_name(
                    "annotations.json",
                    output_file_name_prefix,
                    stem_prefix=output_file_stem_prefix,
                    stem_suffix=output_file_stem_suffix,
                ),
                source_descriptor="annotations.json",
            )
        )

    for image in dataset.images:
        source_descriptor = image.file_name
        if destination_format == SourceFormat.YOLO:
            targets.append(
                PlannedOutputTarget(
                    kind="label",
                    rel_path=label_output_rel_path_for_image(
                        image.file_name,
                        image.image_id,
                        flatten_output_layout=flatten_output_layout,
                        output_file_stem_prefix=output_file_stem_prefix,
                        output_file_stem_suffix=output_file_stem_suffix,
                    ).as_posix(),
                    source_descriptor=source_descriptor,
                )
            )

        if not copy_images:
            continue

        image_target = (
            image_output_rel_path_for_image(
                image.file_name,
                image.image_id,
                flatten_output_layout=flatten_output_layout,
                output_file_stem_prefix=output_file_stem_prefix,
                output_file_stem_suffix=output_file_stem_suffix,
            )
            if destination_format == SourceFormat.YOLO
            else Path(image.file_name)
        )
        targets.append(
            PlannedOutputTarget(
                kind="image",
                rel_path=image_target.as_posix(),
                source_descriptor=source_descriptor,
            )
        )

    return targets


def _ensure_unique_output_targets(
    dataset: AnnotationDataset,
    *,
    destination_format: SourceFormat,
    copy_images: bool,
    flatten_output_layout: bool,
    output_file_name_prefix: str | None,
    output_file_stem_prefix: str | None,
    output_file_stem_suffix: str | None,
) -> None:
    label_targets: dict[str, list[str]] = {}
    image_targets: dict[str, list[str]] = {}

    for target in _collect_planned_output_targets(
        dataset,
        destination_format=destination_format,
        copy_images=copy_images,
        flatten_output_layout=flatten_output_layout,
        output_file_name_prefix=output_file_name_prefix,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    ):
        if target.kind == "label":
            label_targets.setdefault(target.rel_path, []).append(target.source_descriptor)
        elif target.kind == "image":
            image_targets.setdefault(target.rel_path, []).append(target.source_descriptor)

    collisions: list[tuple[str, str, list[str]]] = []
    collisions.extend(
        ("label", output_path, sources)
        for output_path, sources in _find_duplicate_output_targets(label_targets)
    )
    collisions.extend(
        ("image", output_path, sources)
        for output_path, sources in _find_duplicate_output_targets(image_targets)
    )

    if not collisions:
        return

    first_kind, first_path, first_sources = collisions[0]
    suggestion = (
        "Disable shared output layout or adjust the filename prefix/suffix settings so each source file "
        "maps to a unique output path."
        if flatten_output_layout
        else "Adjust the filename prefix/suffix settings so each source file maps to a unique output path."
    )
    raise ConversionError(
        "Output path collision detected; conversion would overwrite files.",
        context={
            "collision_count": str(len(collisions)),
            "first_collision_kind": first_kind,
            "first_collision_path": first_path,
            "first_collision_sources": "; ".join(first_sources[:5]),
            "recommended_fix": suggestion,
        },
    )


def _inspect_existing_output_conflict(
    output_root: Path,
    target: PlannedOutputTarget,
) -> ExistingOutputConflict | None:
    base = output_root.resolve()
    target_path = safe_resolve(output_root, target.rel_path)
    current = target_path

    while True:
        if current.exists():
            if current == target_path:
                return ExistingOutputConflict(
                    kind=target.kind if not current.is_dir() else f"{target.kind}_target_directory",
                    rel_path=target.rel_path,
                    source_descriptor=target.source_descriptor,
                    can_overwrite=not current.is_dir(),
                )
            if not current.is_dir():
                return ExistingOutputConflict(
                    kind=f"{target.kind}_parent_path",
                    rel_path="." if current == base else current.relative_to(base).as_posix(),
                    source_descriptor=target.source_descriptor,
                    can_overwrite=False,
                )
        if current == base:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _collect_existing_output_conflicts(
    dataset: AnnotationDataset,
    *,
    output_root: Path,
    destination_format: SourceFormat,
    copy_images: bool,
    flatten_output_layout: bool,
    output_file_name_prefix: str | None,
    output_file_stem_prefix: str | None,
    output_file_stem_suffix: str | None,
) -> list[ExistingOutputConflict]:
    existing_conflicts: list[ExistingOutputConflict] = []
    seen_conflict_keys: set[tuple[str, str]] = set()

    for target in _collect_planned_output_targets(
        dataset,
        destination_format=destination_format,
        copy_images=copy_images,
        flatten_output_layout=flatten_output_layout,
        output_file_name_prefix=output_file_name_prefix,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    ):
        conflict = _inspect_existing_output_conflict(output_root, target)
        if conflict is None:
            continue
        conflict_key = (conflict.kind, conflict.rel_path)
        if conflict_key in seen_conflict_keys:
            continue
        seen_conflict_keys.add(conflict_key)
        existing_conflicts.append(conflict)

    return existing_conflicts


def _existing_output_conflict_recommendation(
    *,
    conflict_kind: str,
    allow_overwrite: bool,
) -> str:
    if conflict_kind.endswith("_parent_path") or conflict_kind.endswith("_target_directory"):
        return "Remove or rename the conflicting path so the converter can create the required output directories/files."
    if allow_overwrite:
        return "Remove or rename the conflicting path before retrying."
    return "Choose an empty output directory, remove the conflicting files, or enable overwrite and retry."


def _build_output_overwrite_warnings(
    overwrites: list[ExistingOutputConflict],
) -> list[WarningEvent]:
    if not overwrites:
        return []

    overwritten_paths = sorted(item.rel_path for item in overwrites)
    overwritten_kind_counts: dict[str, int] = {}
    for item in overwrites:
        overwritten_kind_counts[item.kind] = overwritten_kind_counts.get(item.kind, 0) + 1

    overwrite_count = len(overwritten_paths)
    file_label = "file" if overwrite_count == 1 else "files"
    context = {
        "overwritten_count": str(overwrite_count),
        "overwritten_paths_json": json.dumps(overwritten_paths),
        "overwritten_kind_counts_json": json.dumps(overwritten_kind_counts, sort_keys=True),
    }
    sample_paths = overwritten_paths[:10]
    if sample_paths:
        context["sample_overwritten_paths_json"] = json.dumps(sample_paths)

    return [
        WarningEvent(
            code="output_files_overwritten",
            message=f"Overwrote {overwrite_count} existing output {file_label} because overwrite was enabled.",
            severity=Severity.WARNING,
            context=context,
        )
    ]


def _ensure_output_targets_do_not_already_exist(
    dataset: AnnotationDataset,
    *,
    output_root: Path,
    destination_format: SourceFormat,
    copy_images: bool,
    flatten_output_layout: bool,
    output_file_name_prefix: str | None,
    output_file_stem_prefix: str | None,
    output_file_stem_suffix: str | None,
    allow_overwrite: bool,
) -> list[WarningEvent]:
    existing_conflicts = _collect_existing_output_conflicts(
        dataset,
        output_root=output_root,
        destination_format=destination_format,
        copy_images=copy_images,
        flatten_output_layout=flatten_output_layout,
        output_file_name_prefix=output_file_name_prefix,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    )
    if not existing_conflicts:
        return []

    blocking_conflicts = [conflict for conflict in existing_conflicts if not conflict.can_overwrite]
    if blocking_conflicts:
        first_conflict = blocking_conflicts[0]
        raise ConversionError(
            "Existing destination path detected; conversion would overwrite or conflict with existing files.",
            context={
                "existing_count": str(len(blocking_conflicts)),
                "first_existing_kind": first_conflict.kind,
                "first_existing_path": first_conflict.rel_path,
                "first_existing_source": first_conflict.source_descriptor,
                "recommended_fix": _existing_output_conflict_recommendation(
                    conflict_kind=first_conflict.kind,
                    allow_overwrite=allow_overwrite,
                ),
            },
        )

    if not allow_overwrite:
        first_conflict = existing_conflicts[0]
        raise ConversionError(
            "Existing destination path detected; conversion would overwrite or conflict with existing files.",
            context={
                "existing_count": str(len(existing_conflicts)),
                "first_existing_kind": first_conflict.kind,
                "first_existing_path": first_conflict.rel_path,
                "first_existing_source": first_conflict.source_descriptor,
                "recommended_fix": _existing_output_conflict_recommendation(
                    conflict_kind=first_conflict.kind,
                    allow_overwrite=allow_overwrite,
                ),
            },
        )

    return _build_output_overwrite_warnings(existing_conflicts)


def _save_resized_image(
    source_path: Path,
    destination_path: Path,
    *,
    width: int,
    height: int,
) -> None:
    with Image.open(source_path) as opened:
        resized = opened.resize((width, height), _IMAGE_RESAMPLING_LANCZOS)
        suffix = destination_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        ensure_directory(destination_path.parent)
        resized.save(destination_path)


def _resize_image_in_place(
    image_path: Path,
    *,
    width: int,
    height: int,
) -> None:
    with Image.open(image_path) as opened:
        resized = opened.resize((width, height), _IMAGE_RESAMPLING_LANCZOS)
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        resized.save(image_path)


def _copy_images_to_output(
    *,
    dataset: ValidationOutcome,
    output_dataset: AnnotationDataset,
    destination_format: SourceFormat,
    input_root: Path,
    output_root: Path,
    flatten_output_layout: bool = False,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
    progress_callback: ConversionProgressCallback | None = None,
    start_percent: int = 80,
    end_percent: int = 96,
) -> list[WarningEvent]:
    warnings: list[WarningEvent] = []
    output_images_by_id = {image.image_id: image for image in output_dataset.images}
    source_images = [
        image for image in sorted(dataset.dataset.images, key=lambda item: item.image_id) if image.image_id in output_images_by_id
    ]

    if (
        dataset.dataset.source_format == SourceFormat.VIDEO_BBOX
        or dataset.dataset.source_metadata.details.get("media_kind") == "video_collection"
    ):
        _emit_progress(
            progress_callback,
            message="Copying extracted video frames...",
            percent=start_percent,
        )
        output_images = [output_images_by_id[image.image_id] for image in source_images]
        if destination_format == SourceFormat.YOLO:
            output_images = [
                image.model_copy(
                    update={
                        "file_name": image_output_rel_path_for_image(
                            image.file_name,
                            image.image_id,
                            flatten_output_layout=flatten_output_layout,
                            output_file_stem_prefix=output_file_stem_prefix,
                            output_file_stem_suffix=output_file_stem_suffix,
                        ).as_posix()
                    }
                )
                for image in output_dataset.images
            ]
        materialize_video_bbox_frames(
            dataset_root=input_root,
            images=source_images,
            output_root=output_root,
            output_images=output_images,
            progress_callback=(
                lambda source_name, completed_frames, total_frames: _emit_progress(
                    progress_callback,
                    message=(
                        f"Materializing video frames from {source_name} "
                        f"({completed_frames}/{total_frames})"
                    ),
                    percent=_interpolate_progress(
                        start_percent=start_percent,
                        end_percent=end_percent,
                        completed_units=completed_frames,
                        total_units=total_frames,
                    ),
                )
            ),
        )
        source_images_by_id = {image.image_id: image for image in source_images}
        for output_image in output_images:
            source_image = source_images_by_id.get(output_image.image_id)
            if source_image is None:
                continue
            if (source_image.width, source_image.height) == (output_image.width, output_image.height):
                continue
            destination_path = safe_resolve(output_root, output_image.file_name)
            _resize_image_in_place(
                destination_path,
                width=output_image.width,
                height=output_image.height,
            )
        _emit_progress(
            progress_callback,
            message="Finished copying extracted video frames.",
            percent=end_percent,
        )
        return warnings

    total_images = len(source_images)
    update_every = max(total_images // 100, 1) if total_images else 1
    _emit_progress(progress_callback, message="Copying images...", percent=start_percent)

    for index, image in enumerate(source_images, start=1):
        output_image = output_images_by_id[image.image_id]
        try:
            source_path = safe_resolve(input_root, image.file_name)
            destination_rel = (
                image_output_rel_path_for_image(
                    output_image.file_name,
                    output_image.image_id,
                    flatten_output_layout=flatten_output_layout,
                    output_file_stem_prefix=output_file_stem_prefix,
                    output_file_stem_suffix=output_file_stem_suffix,
                )
                if destination_format == SourceFormat.YOLO
                else Path(output_image.file_name)
            )
            destination_path = safe_resolve(output_root, destination_rel.as_posix())
        except Exception as exc:
            warnings.append(
                WarningEvent(
                    code="image_copy_path_error",
                    message=f"Unable to resolve image copy path for {image.file_name}",
                    severity=Severity.WARNING,
                    context={"reason": str(exc)},
                )
            )
        else:
            if not source_path.exists() or not source_path.is_file():
                warnings.append(
                    WarningEvent(
                        code="image_copy_source_missing",
                        message=f"Image source missing for copy: {image.file_name}",
                        severity=Severity.WARNING,
                        context={"source_path": str(source_path)},
                    )
                )
            else:
                ensure_directory(destination_path.parent)
                if (image.width, image.height) != (output_image.width, output_image.height):
                    _save_resized_image(
                        source_path,
                        destination_path,
                        width=output_image.width,
                        height=output_image.height,
                    )
                else:
                    copy2(source_path, destination_path)

        if index % update_every == 0 or index == total_images:
            _emit_progress(
                progress_callback,
                message=f"Copying images... ({index}/{total_images})",
                percent=_interpolate_progress(
                    start_percent=start_percent,
                    end_percent=end_percent,
                    completed_units=index,
                    total_units=max(total_images, 1),
                ),
            )

    return warnings


def execute_conversion(
    request: ConvertRequest,
    *,
    lock_manager: OutputPathLockManager | None = None,
    progress_callback: ConversionProgressCallback | None = None,
) -> ConvertResult:
    lock_manager = lock_manager or OutputPathLockManager()
    _emit_progress(progress_callback, message="Starting conversion...", percent=5)

    src_format = request.src_format
    inference_warnings: list[WarningEvent] = []
    validation_warnings: list[WarningEvent] = []
    size_gate_warnings: list[WarningEvent] = []
    copy_warnings: list[WarningEvent] = []
    overwrite_warnings: list[WarningEvent] = []
    dropped_annotations: list[DroppedAnnotationModel] = []
    if src_format == SourceFormat.AUTO:
        _emit_progress(progress_callback, message="Inferring source format...", percent=12)
        inference = infer_format(request.input_path, force=request.force_infer)
        inference_warnings = inference.warnings
        if inference.predicted_format not in {
            SourceFormat.COCO,
            SourceFormat.CUSTOM,
            SourceFormat.KITWARE,
            SourceFormat.MATLAB_GROUND_TRUTH,
            SourceFormat.VOC,
            SourceFormat.VIDEO_BBOX,
            SourceFormat.YOLO,
        }:
            raise ConversionError(
                "Unable to resolve source format for conversion",
                context={"predicted": inference.predicted_format.value},
            )
        src_format = inference.predicted_format

    resolved_destination = request.dst_format or src_format

    _emit_progress(progress_callback, message="Validating dataset...", percent=28)
    validation_load_percent_end = 38
    validation_annotation_percent_end = 44
    validation = validate_dataset(
        request.input_path,
        source_format=src_format,
        policy=ValidationPolicy.for_mode(
            request.validation_mode,
            invalid_annotation_action=request.permissive_invalid_annotation_action,
            correct_out_of_frame_bboxes=request.correct_out_of_frame_bboxes,
            out_of_frame_tolerance_px=request.out_of_frame_tolerance_px,
        ),
        load_progress_callback=(
            lambda completed_files, total_files: _emit_progress(
                progress_callback,
                message=f"Loading source files... ({completed_files}/{total_files})",
                percent=_interpolate_progress(
                    start_percent=28,
                    end_percent=validation_load_percent_end,
                    completed_units=completed_files,
                    total_units=max(total_files, 1),
                ),
            )
        ),
        annotation_progress_callback=(
            lambda completed_annotations, total_annotations: _emit_progress(
                progress_callback,
                message=f"Validating annotations... ({completed_annotations}/{total_annotations})",
                percent=_interpolate_progress(
                    start_percent=validation_load_percent_end,
                    end_percent=validation_annotation_percent_end,
                    completed_units=completed_annotations,
                    total_units=max(total_annotations, 1),
                ),
            )
        ),
        input_path_include_substring=request.input_path_include_substring,
        input_path_exclude_substring=request.input_path_exclude_substring,
    )
    validation_warnings = validation.warnings
    dropped_annotations.extend(validation.dropped_annotations)

    if (
        request.max_image_longest_edge_px > 0
        and request.oversize_image_action == OversizeImageAction.DOWNSCALE
        and not request.copy_images
        and not request.dry_run
    ):
        raise ConversionError(
            "Downscaling oversized images requires 'copy_images' unless this is a dry run."
        )
    if (
        request.min_image_longest_edge_px > 0
        and request.max_image_longest_edge_px > 0
        and request.min_image_longest_edge_px > request.max_image_longest_edge_px
    ):
        raise ConversionError("Minimum image size gate cannot exceed the maximum image size gate.")

    remap_result: RemapResult | None = None
    working_dataset = validation.dataset
    working_dataset, size_gate_warnings, size_gate_dropped_annotations = _apply_image_size_gate(
        working_dataset,
        min_image_longest_edge_px=request.min_image_longest_edge_px,
        max_image_longest_edge_px=request.max_image_longest_edge_px,
        oversize_image_action=request.oversize_image_action,
    )
    dropped_annotations.extend(size_gate_dropped_annotations)
    if request.class_map:
        _emit_progress(progress_callback, message="Applying class remap...", percent=45)
        remap_result = apply_class_remap(
            working_dataset,
            request.class_map,
            policy=RemapPolicy(unmapped_policy=request.unmapped_policy),
        )
        working_dataset = remap_result.dataset
        dropped_annotations.extend(remap_result.dropped_annotations)

    output_file_name_prefix = (
        sanitize_output_file_name_prefix(request.output_file_name_prefix)
        if request.output_file_name_prefix
        else None
    )
    output_file_stem_prefix = (
        sanitize_output_file_stem_affix(request.output_file_stem_prefix)
        if request.output_file_stem_prefix
        else None
    )
    output_file_stem_suffix = (
        sanitize_output_file_stem_affix(request.output_file_stem_suffix)
        if request.output_file_stem_suffix
        else None
    )
    flatten_output_layout = (
        request.flatten_output_layout
        and resolved_destination == SourceFormat.YOLO
    )
    output_dataset = (
        _dataset_with_prefixed_output_file_names(working_dataset, output_file_name_prefix)
        if output_file_name_prefix
        else working_dataset
    )
    if resolved_destination == SourceFormat.COCO:
        output_dataset = _dataset_with_output_file_stem_affixes(
            output_dataset,
            stem_prefix=output_file_stem_prefix,
            stem_suffix=output_file_stem_suffix,
        )

    _ensure_unique_output_targets(
        output_dataset,
        destination_format=resolved_destination,
        copy_images=request.copy_images,
        flatten_output_layout=flatten_output_layout,
        output_file_name_prefix=output_file_name_prefix,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    )

    contention_events: list[ContentionEventModel] = []
    output_artifacts: list[Path] = []

    if request.output_path and not request.dry_run:
        _emit_progress(progress_callback, message="Acquiring output lock...", percent=58)
        events = lock_manager.acquire(request.output_path, request.run_id)
        contention_events = [ContentionEventModel.model_validate(event.model_dump(mode="python")) for event in events]
        overwrite_warnings = _ensure_output_targets_do_not_already_exist(
            output_dataset,
            output_root=request.output_path,
            destination_format=resolved_destination,
            copy_images=request.copy_images,
            flatten_output_layout=flatten_output_layout,
            output_file_name_prefix=output_file_name_prefix,
            output_file_stem_prefix=output_file_stem_prefix,
            output_file_stem_suffix=output_file_stem_suffix,
            allow_overwrite=request.allow_overwrite,
        )

        destination = resolved_destination
        _emit_progress(
            progress_callback,
            message=f"Writing {destination.value.upper()} output...",
            percent=72,
        )
        if destination == SourceFormat.YOLO:
            output_artifacts.append(
                write_yolo_dataset(
                    output_dataset,
                    request.output_path,
                    classes_file_name=_artifact_file_name(
                        "classes.txt",
                        output_file_name_prefix,
                        stem_prefix=output_file_stem_prefix,
                        stem_suffix=output_file_stem_suffix,
                    ),
                    flatten_output_layout=flatten_output_layout,
                    output_file_stem_prefix=output_file_stem_prefix,
                    output_file_stem_suffix=output_file_stem_suffix,
                    progress_callback=(
                        lambda completed, total: _emit_progress(
                            progress_callback,
                            message=f"Writing YOLO labels... ({completed}/{total})",
                            percent=_interpolate_progress(
                                start_percent=72,
                                end_percent=80 if request.copy_images else 88,
                                completed_units=completed,
                                total_units=total,
                            ),
                        )
                    ),
                )
            )
        elif destination == SourceFormat.COCO:
            output_artifacts.append(
                write_coco_dataset(
                    output_dataset,
                    request.output_path,
                    annotations_file_name=_artifact_file_name(
                        "annotations.json",
                        output_file_name_prefix,
                        stem_prefix=output_file_stem_prefix,
                        stem_suffix=output_file_stem_suffix,
                    ),
                )
            )
        else:
            raise ConversionError("Unsupported destination format", context={"dst": str(destination)})

        if request.copy_images:
            copy_warnings = _copy_images_to_output(
                dataset=validation,
                output_dataset=output_dataset,
                destination_format=destination,
                input_root=request.input_path,
                output_root=request.output_path,
                flatten_output_layout=flatten_output_layout,
                output_file_stem_prefix=output_file_stem_prefix,
                output_file_stem_suffix=output_file_stem_suffix,
                progress_callback=progress_callback,
            )

        lock_manager.mark_completed(request.output_path, request.run_id)
    else:
        _emit_progress(progress_callback, message="Preparing run report...", percent=88)

    dropped = len(dropped_annotations)
    unmapped = remap_result.unmapped if remap_result else 0

    _emit_progress(progress_callback, message="Finalizing run report...", percent=98)
    report = RunReportModel(
        run_id=request.run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path=str(request.input_path),
        output_path=str(request.output_path) if request.output_path else None,
        src_format=_format_src_for_report(request.src_format),
        dst_format=_format_dst_for_report(request.dst_format),
        summary_counts=SummaryCountsModel(
            images=len(output_dataset.images),
            annotations_in=len(validation.dataset.annotations),
            annotations_out=len(working_dataset.annotations),
            dropped=dropped,
            unmapped=unmapped,
            invalid=validation.summary.invalid_annotations,
            skipped=max(0, len(validation.dataset.images) - len(output_dataset.images)),
        ),
        warnings=_build_warning_models(
            inference_warnings + validation_warnings + size_gate_warnings + overwrite_warnings + copy_warnings
        ),
        contention_events=contention_events,
        provenance=[],
    )
    _emit_progress(progress_callback, message="Conversion complete.", percent=100)

    return ConvertResult(
        report=report,
        output_artifacts=output_artifacts,
        validation=validation,
        remap=remap_result,
        output_dataset=output_dataset,
        dropped_annotations=dropped_annotations,
    )


def execute_dry_run(
    request: ConvertRequest,
    *,
    lock_manager: OutputPathLockManager | None = None,
    progress_callback: ConversionProgressCallback | None = None,
) -> ConvertResult:
    dry_request = ConvertRequest(
        run_id=request.run_id,
        input_path=request.input_path,
        output_path=request.output_path,
        src_format=request.src_format,
        dst_format=request.dst_format,
        class_map=request.class_map,
        unmapped_policy=request.unmapped_policy,
        dry_run=True,
        force_infer=request.force_infer,
        copy_images=request.copy_images,
        allow_overwrite=request.allow_overwrite,
        input_path_include_substring=request.input_path_include_substring,
        input_path_exclude_substring=request.input_path_exclude_substring,
        output_file_name_prefix=request.output_file_name_prefix,
        output_file_stem_prefix=request.output_file_stem_prefix,
        output_file_stem_suffix=request.output_file_stem_suffix,
        flatten_output_layout=request.flatten_output_layout,
        validation_mode=request.validation_mode,
        permissive_invalid_annotation_action=request.permissive_invalid_annotation_action,
        correct_out_of_frame_bboxes=request.correct_out_of_frame_bboxes,
        out_of_frame_tolerance_px=request.out_of_frame_tolerance_px,
        min_image_longest_edge_px=request.min_image_longest_edge_px,
        max_image_longest_edge_px=request.max_image_longest_edge_px,
        oversize_image_action=request.oversize_image_action,
    )
    return execute_conversion(
        dry_request,
        lock_manager=lock_manager,
        progress_callback=progress_callback,
    )


def verify_known_bbox_dry_run_sample(manifest_path: Path) -> DryRunVerificationResult:
    return verify_dry_run_manifest(manifest_path)
