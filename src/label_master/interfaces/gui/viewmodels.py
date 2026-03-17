from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    InferencePolicy,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.services.convert_service import (
    ConvertRequest,
    ConvertResult,
    execute_conversion,
)
from label_master.core.services.infer_service import infer_format
from label_master.core.services.validate_service import validate_dataset
from label_master.infra.config import load_mapping_file
from label_master.infra.reporting import generate_run_id
from label_master.reports.schemas import RunConfigModel


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


def preview_dataset_view(input_path: Path, *, source_format: str) -> DatasetPreviewViewModel:
    resolved_format = SourceFormat(source_format)
    if resolved_format not in {SourceFormat.COCO, SourceFormat.YOLO}:
        raise ValueError("Preview source format must be coco or yolo")

    validation = validate_dataset(
        input_path,
        source_format=resolved_format,
        policy=ValidationPolicy.for_mode(ValidationMode.PERMISSIVE),
    )
    dataset = validation.dataset

    warnings: list[str] = []
    if validation.summary.invalid_annotations:
        warnings.append(
            f"Preview loaded with {validation.summary.invalid_annotations} invalid annotation(s)."
        )

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


def infer_view(input_path: Path, *, sample_limit: int = 500) -> InferenceViewModel:
    result = infer_format(input_path, policy=InferencePolicy(sample_limit=sample_limit), force=True)
    return InferenceViewModel(
        predicted_format=result.predicted_format.value,
        confidence=result.confidence,
        candidates=[(candidate.format.value, candidate.score) for candidate in result.candidates],
        warnings=[warning.message for warning in result.warnings],
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
        )
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
) -> RunConfigModel:
    return RunConfigModel(
        run_id=run_id,
        mode="convert",
        input_path=str(input_path),
        output_path=str(output_path),
        src_format=src,  # type: ignore[arg-type]
        dst_format=dst,  # type: ignore[arg-type]
        mapping_file=str(map_path) if map_path else None,
        unmapped_policy=unmapped_policy,  # type: ignore[arg-type]
        dry_run=dry_run,
        created_at=datetime.now(UTC),
    )
