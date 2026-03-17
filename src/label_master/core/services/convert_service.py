from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from typing import Literal

from label_master.adapters.coco.writer import write_coco_dataset
from label_master.adapters.yolo.writer import write_yolo_dataset
from label_master.core.domain.entities import Severity, SourceFormat, WarningEvent
from label_master.core.domain.policies import (
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


@dataclass(frozen=True)
class ConvertResult:
    report: RunReportModel
    output_artifacts: list[Path]
    validation: ValidationOutcome
    remap: RemapResult | None


RunSrcFormatLiteral = Literal["auto", "coco", "yolo"]
RunDstFormatLiteral = Literal["coco", "yolo"]


def _format_src_for_report(source_format: SourceFormat | None) -> RunSrcFormatLiteral | None:
    if source_format is None:
        return None
    if source_format == SourceFormat.AUTO:
        return "auto"
    if source_format == SourceFormat.COCO:
        return "coco"
    if source_format == SourceFormat.YOLO:
        return "yolo"
    return None


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


def _copy_images_to_output(
    *,
    dataset: ValidationOutcome,
    input_root: Path,
    output_root: Path,
) -> list[WarningEvent]:
    warnings: list[WarningEvent] = []

    for image in sorted(dataset.dataset.images, key=lambda item: item.image_id):
        try:
            source_path = safe_resolve(input_root, image.file_name)
            destination_path = safe_resolve(output_root, image.file_name)
        except Exception as exc:
            warnings.append(
                WarningEvent(
                    code="image_copy_path_error",
                    message=f"Unable to resolve image copy path for {image.file_name}",
                    severity=Severity.WARNING,
                    context={"reason": str(exc)},
                )
            )
            continue

        if not source_path.exists() or not source_path.is_file():
            warnings.append(
                WarningEvent(
                    code="image_copy_source_missing",
                    message=f"Image source missing for copy: {image.file_name}",
                    severity=Severity.WARNING,
                    context={"source_path": str(source_path)},
                )
            )
            continue

        ensure_directory(destination_path.parent)
        copy2(source_path, destination_path)

    return warnings


def execute_conversion(
    request: ConvertRequest,
    *,
    lock_manager: OutputPathLockManager | None = None,
) -> ConvertResult:
    lock_manager = lock_manager or OutputPathLockManager()

    src_format = request.src_format
    inference_warnings: list[WarningEvent] = []
    copy_warnings: list[WarningEvent] = []
    if src_format == SourceFormat.AUTO:
        inference = infer_format(request.input_path, force=request.force_infer)
        inference_warnings = inference.warnings
        if inference.predicted_format not in {SourceFormat.COCO, SourceFormat.YOLO}:
            raise ConversionError(
                "Unable to resolve source format for conversion",
                context={"predicted": inference.predicted_format.value},
            )
        src_format = inference.predicted_format

    validation = validate_dataset(
        request.input_path,
        source_format=src_format,
        policy=ValidationPolicy(mode=ValidationMode.STRICT),
    )

    remap_result: RemapResult | None = None
    working_dataset = validation.dataset
    if request.class_map:
        remap_result = apply_class_remap(
            validation.dataset,
            request.class_map,
            policy=RemapPolicy(unmapped_policy=request.unmapped_policy),
        )
        working_dataset = remap_result.dataset

    contention_events: list[ContentionEventModel] = []
    output_artifacts: list[Path] = []

    if request.output_path and not request.dry_run:
        events = lock_manager.acquire(request.output_path, request.run_id)
        contention_events = [ContentionEventModel.model_validate(event.model_dump(mode="python")) for event in events]

        destination = request.dst_format or src_format
        if destination == SourceFormat.YOLO:
            output_artifacts.append(write_yolo_dataset(working_dataset, request.output_path))
        elif destination == SourceFormat.COCO:
            output_artifacts.append(write_coco_dataset(working_dataset, request.output_path))
        else:
            raise ConversionError("Unsupported destination format", context={"dst": str(destination)})

        if request.copy_images:
            copy_warnings = _copy_images_to_output(
                dataset=validation,
                input_root=request.input_path,
                output_root=request.output_path,
            )

        lock_manager.mark_completed(request.output_path, request.run_id)

    dropped = remap_result.dropped if remap_result else 0
    unmapped = remap_result.unmapped if remap_result else 0

    report = RunReportModel(
        run_id=request.run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path=str(request.input_path),
        output_path=str(request.output_path) if request.output_path else None,
        src_format=_format_src_for_report(request.src_format),
        dst_format=_format_dst_for_report(request.dst_format),
        summary_counts=SummaryCountsModel(
            images=len(validation.dataset.images),
            annotations_in=len(validation.dataset.annotations),
            annotations_out=len(working_dataset.annotations),
            dropped=dropped,
            unmapped=unmapped,
            invalid=validation.summary.invalid_annotations,
            skipped=0,
        ),
        warnings=_build_warning_models(inference_warnings + copy_warnings),
        contention_events=contention_events,
        provenance=[],
    )

    return ConvertResult(
        report=report,
        output_artifacts=output_artifacts,
        validation=validation,
        remap=remap_result,
    )


def execute_dry_run(
    request: ConvertRequest,
    *,
    lock_manager: OutputPathLockManager | None = None,
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
    )
    return execute_conversion(dry_request, lock_manager=lock_manager)


def verify_known_bbox_dry_run_sample(manifest_path: Path) -> DryRunVerificationResult:
    return verify_dry_run_manifest(manifest_path)
