from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import typer

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import (
    InferencePolicy,
    OversizeImageAction,
    UnmappedPolicy,
    ValidationMode,
    ValidationPolicy,
)
from label_master.core.domain.value_objects import (
    ConfigurationError,
    ConversionError,
    ImportError,
    InferenceError,
    ValidationError,
)
from label_master.core.services.convert_service import (
    ConvertRequest,
    execute_conversion,
    verify_known_bbox_dry_run_sample,
)
from label_master.core.services.import_service import import_dataset
from label_master.core.services.infer_service import infer_format
from label_master.core.services.validate_service import validate_dataset
from label_master.infra.config import load_mapping_file
from label_master.infra.logging import setup_logging
from label_master.infra.reporting import generate_run_id, persist_run_artifacts
from label_master.reports.schemas import RunConfigModel, WarningEventModel

app = typer.Typer(help="Bounding-box annotation conversion toolkit")

RunSrcFormat = Literal["auto", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
RunDstFormat = Literal["coco", "yolo"]
ProviderName = Literal["kaggle", "roboflow", "github", "direct_url"]


@dataclass
class CLIState:
    report_path: Path | None


def _parse_source_format(value: str) -> SourceFormat:
    try:
        return SourceFormat(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unsupported format value: {value}") from exc


def _emit_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _artifact_dir(path: Path | None) -> Path:
    return path or Path("reports")


def _to_run_src_format(source_format: SourceFormat) -> RunSrcFormat:
    return source_format.value  # type: ignore[return-value]


def _to_run_dst_format(source_format: SourceFormat) -> RunDstFormat:
    if source_format == SourceFormat.COCO:
        return "coco"
    if source_format == SourceFormat.YOLO:
        return "yolo"
    raise ConfigurationError(f"Unsupported destination format for run artifact: {source_format.value}")


def _parse_provider_name(provider: str) -> ProviderName:
    normalized = provider.lower()
    if normalized in {"kaggle", "roboflow", "github", "direct_url"}:
        return normalized  # type: ignore[return-value]
    raise ConfigurationError(f"Unsupported provider: {provider}")


@app.callback()
def callback(  # noqa: PLR0913
    ctx: typer.Context,
    config: Path | None = typer.Option(None, help="Path to config file"),
    log_level: str = typer.Option("info", help="Log level: debug|info|warn|error"),
    log_file: Path | None = typer.Option(None, help="Optional log file path"),
    report_path: Path | None = typer.Option(None, help="Directory for run artifacts"),
) -> None:
    del config
    setup_logging(log_level.upper(), log_file)
    ctx.obj = CLIState(report_path=report_path)


@app.command("infer")
def infer_command(
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=False, dir_okay=True),
    json_output: bool = typer.Option(False, "--json"),
    sample_limit: int = typer.Option(500, "--sample-limit", min=1),
    force: bool = typer.Option(False, "--force"),
) -> None:
    run_id = generate_run_id("infer")
    try:
        result = infer_format(input_path, policy=InferencePolicy(sample_limit=sample_limit), force=force)
    except InferenceError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc

    payload = result.model_dump(mode="json")
    if json_output:
        _emit_json(payload)
    else:
        typer.echo(f"predicted_format={result.predicted_format.value} confidence={result.confidence:.3f}")

    state: CLIState = ctx.obj
    config = RunConfigModel(
        run_id=run_id,
        mode="infer",
        input_path=str(input_path),
        src_format="auto",
        created_at=datetime.now(UTC),
    )
    from label_master.reports.schemas import RunReportModel, SummaryCountsModel

    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path=str(input_path),
        summary_counts=SummaryCountsModel(
            images=0,
            annotations_in=0,
            annotations_out=0,
            dropped=0,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
    )
    persist_run_artifacts(
        _artifact_dir(state.report_path),
        run_id,
        config,
        report,
    )


@app.command("validate")
def validate_command(
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=False, dir_okay=True),
    source_format: str = typer.Option("auto", "--format"),
    strict: bool = typer.Option(False, "--strict"),
    permissive: bool = typer.Option(False, "--permissive"),
) -> None:
    if strict and permissive:
        typer.echo("Cannot set both --strict and --permissive", err=True)
        raise typer.Exit(code=5)

    mode = ValidationMode.PERMISSIVE if permissive else ValidationMode.STRICT
    run_id = generate_run_id("validate")

    try:
        outcome = validate_dataset(
            input_path,
            source_format=_parse_source_format(source_format),
            policy=ValidationPolicy.for_mode(mode),
        )
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=5) from exc

    typer.echo(
        f"valid={outcome.summary.valid} invalid_annotations={outcome.summary.invalid_annotations} "
        f"format={outcome.inferred_format.value}"
    )
    for warning in outcome.warnings:
        typer.echo(f"warning={warning.message}")

    state: CLIState = ctx.obj
    parsed_source = _parse_source_format(source_format) if source_format != "auto" else SourceFormat.AUTO
    config = RunConfigModel(
        run_id=run_id,
        mode="validate",
        input_path=str(input_path),
        src_format=_to_run_src_format(parsed_source),
        created_at=datetime.now(UTC),
    )
    from label_master.reports.schemas import RunReportModel, SummaryCountsModel

    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path=str(input_path),
        src_format=_to_run_src_format(outcome.inferred_format),
        summary_counts=SummaryCountsModel(
            images=len(outcome.dataset.images),
            annotations_in=len(outcome.dataset.annotations),
            annotations_out=len(outcome.dataset.annotations),
            dropped=0,
            unmapped=0,
            invalid=outcome.summary.invalid_annotations,
            skipped=0,
        ),
        warnings=[
            WarningEventModel.model_validate(warning.model_dump(mode="python"))
            for warning in outcome.warnings
        ],
    )
    persist_run_artifacts(
        _artifact_dir(state.report_path),
        run_id,
        config,
        report,
        dropped_annotations=outcome.dropped_annotations,
    )


@app.command("convert")
def convert_command(  # noqa: PLR0913
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Option(..., "--output"),
    src: str = typer.Option("auto", "--src"),
    dst: str = typer.Option(..., "--dst"),
    map_path: Path | None = typer.Option(None, "--map"),
    unmapped_policy: str = typer.Option("error", "--unmapped-policy"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    copy_images: bool = typer.Option(False, "--copy-images/--no-copy-images"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite/--no-allow-overwrite"),
    input_path_include_substring: str | None = typer.Option(None, "--input-path-include-substring"),
    input_path_exclude_substring: str | None = typer.Option(None, "--input-path-exclude-substring"),
    min_image_longest_edge_px: int = typer.Option(0, "--min-image-longest-edge-px", min=0),
    max_image_longest_edge_px: int = typer.Option(0, "--max-image-longest-edge-px", min=0),
    oversize_image_action: str = typer.Option("ignore", "--oversize-image-action"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    run_id = generate_run_id("convert")

    try:
        source_format = _parse_source_format(src)
        destination_format = _parse_source_format(dst)
        if destination_format not in {SourceFormat.COCO, SourceFormat.YOLO}:
            raise ConfigurationError("--dst must be one of coco|yolo")

        policy = UnmappedPolicy(unmapped_policy)
        class_map = load_mapping_file(map_path) if map_path else {}

        result = execute_conversion(
            ConvertRequest(
                run_id=run_id,
                input_path=input_path,
                output_path=output_path,
                src_format=source_format,
                dst_format=destination_format,
                class_map=class_map,
                unmapped_policy=policy,
                dry_run=dry_run,
                force_infer=force,
                copy_images=copy_images,
                allow_overwrite=allow_overwrite,
                input_path_include_substring=input_path_include_substring,
                input_path_exclude_substring=input_path_exclude_substring,
                min_image_longest_edge_px=min_image_longest_edge_px,
                max_image_longest_edge_px=max_image_longest_edge_px,
                oversize_image_action=OversizeImageAction(oversize_image_action),
            )
        )
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except InferenceError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc
    except (ConversionError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=4) from exc
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=5) from exc

    typer.echo(
        "conversion_complete "
        f"annotations_in={result.report.summary_counts.annotations_in} "
        f"annotations_out={result.report.summary_counts.annotations_out} dry_run={dry_run}"
    )

    state: CLIState = ctx.obj
    config = RunConfigModel(
        run_id=run_id,
        mode="convert",
        input_path=str(input_path),
        output_path=str(output_path),
        src_format=_to_run_src_format(source_format),
        dst_format=_to_run_dst_format(destination_format),
        mapping_file=str(map_path) if map_path else None,
        unmapped_policy=policy.value,
        dry_run=dry_run,
        allow_overwrite=allow_overwrite,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        min_image_longest_edge_px=min_image_longest_edge_px,
        max_image_longest_edge_px=max_image_longest_edge_px,
        oversize_image_action=oversize_image_action,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )
    persist_run_artifacts(
        _artifact_dir(state.report_path),
        run_id,
        config,
        result.report,
        dropped_annotations=result.dropped_annotations,
    )


@app.command("remap")
def remap_command(
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Option(..., "--output"),
    source_format: str = typer.Option(..., "--format"),
    map_path: Path = typer.Option(..., "--map", exists=True, file_okay=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
    copy_images: bool = typer.Option(False, "--copy-images/--no-copy-images"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite/--no-allow-overwrite"),
    input_path_include_substring: str | None = typer.Option(None, "--input-path-include-substring"),
    input_path_exclude_substring: str | None = typer.Option(None, "--input-path-exclude-substring"),
) -> None:
    run_id = generate_run_id("remap")

    try:
        src_format = _parse_source_format(source_format)
        class_map = load_mapping_file(map_path)

        result = execute_conversion(
            ConvertRequest(
                run_id=run_id,
                input_path=input_path,
                output_path=output_path,
                src_format=src_format,
                dst_format=src_format,
                class_map=class_map,
                unmapped_policy=UnmappedPolicy.ERROR,
                dry_run=dry_run,
                copy_images=copy_images,
                allow_overwrite=allow_overwrite,
                input_path_include_substring=input_path_include_substring,
                input_path_exclude_substring=input_path_exclude_substring,
            )
        )
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except (ConversionError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=4) from exc
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=5) from exc

    typer.echo(
        f"remap_complete annotations_in={result.report.summary_counts.annotations_in} "
        f"annotations_out={result.report.summary_counts.annotations_out} dry_run={dry_run}"
    )

    state: CLIState = ctx.obj
    config = RunConfigModel(
        run_id=run_id,
        mode="remap",
        input_path=str(input_path),
        output_path=str(output_path),
        src_format=_to_run_src_format(src_format),
        mapping_file=str(map_path),
        dry_run=dry_run,
        allow_overwrite=allow_overwrite,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        created_at=datetime.now(UTC),
    )
    persist_run_artifacts(
        _artifact_dir(state.report_path),
        run_id,
        config,
        result.report,
        dropped_annotations=result.dropped_annotations,
    )


@app.command("import")
def import_command(
    ctx: typer.Context,
    provider: str = typer.Option(..., "--provider"),
    source_ref: str = typer.Option(..., "--source-ref"),
    output_path: Path = typer.Option(..., "--output"),
) -> None:
    run_id = generate_run_id("import")
    try:
        provider_name = _parse_provider_name(provider)
        result = import_dataset(
            provider=provider_name,
            source_ref=source_ref,
            output_path=output_path,
            run_id=run_id,
        )
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=5) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=5) from exc

    typer.echo(f"import_complete provider={provider} path={result.local_path}")

    state: CLIState = ctx.obj
    config = RunConfigModel(
        run_id=run_id,
        mode="import",
        input_path=source_ref,
        output_path=str(output_path),
        src_format="auto",
        provider=provider_name,
        source_ref=source_ref,
        created_at=datetime.now(UTC),
    )
    persist_run_artifacts(_artifact_dir(state.report_path), run_id, config, result.report)


@app.command("verify-dry-run-manifest")
def verify_dry_run_manifest_command(
    manifest: Path = typer.Option(..., "--manifest", exists=True, file_okay=True, dir_okay=False)
) -> None:
    result = verify_known_bbox_dry_run_sample(manifest)
    if not result.success:
        for message in result.diagnostics:
            typer.echo(message, err=True)
        raise typer.Exit(code=result.expected_exit_code or 2)
    typer.echo(f"dry_run_manifest_verified sample={result.sample_id}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
