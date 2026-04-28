from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from label_master.infra.filesystem import atomic_write_json, ensure_directory
from label_master.reports.schemas import DroppedAnnotationModel, RunConfigModel, RunReportModel


def utc_now() -> datetime:
    return datetime.now(UTC)


def generate_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def write_run_config(path: Path, config: RunConfigModel | dict[str, Any]) -> Path:
    payload = config.model_dump(mode="json") if isinstance(config, RunConfigModel) else dict(config)
    atomic_write_json(path, payload)
    return path


def write_run_report(path: Path, report: RunReportModel | dict[str, Any]) -> Path:
    payload = report.model_dump(mode="json") if isinstance(report, RunReportModel) else dict(report)
    atomic_write_json(path, payload)
    return path


def _normalize_dropped_annotations(
    dropped_annotations: list[DroppedAnnotationModel] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in dropped_annotations or []:
        if isinstance(item, DroppedAnnotationModel):
            normalized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _build_file_infractions(dropped_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    for item in dropped_annotations:
        source_file = item.get("source_file")
        if not isinstance(source_file, str) or not source_file.strip():
            source_file = None
        image_file = item.get("image_file")
        if not isinstance(image_file, str) or not image_file.strip():
            image_file = source_file
        image_id = item.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            image_id = None
        key = (image_file, None if image_file is not None else image_id)
        grouped.setdefault(key, []).append(item)

    file_infractions: list[dict[str, Any]] = []
    for (image_file, image_id), items in sorted(grouped.items(), key=lambda item: ((item[0][0] or ""), (item[0][1] or ""))):
        file_infractions.append(
            {
                "file": image_file,
                "image_id": image_id,
                "infraction_count": len(items),
                "infractions": items,
            }
        )
    return file_infractions


def build_run_warnings_payload(
    report: RunReportModel | dict[str, Any],
    *,
    dropped_annotations: list[DroppedAnnotationModel] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = report.model_dump(mode="json") if isinstance(report, RunReportModel) else dict(report)
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warning_messages = [
        str(item.get("message", "")).strip()
        for item in warnings
        if isinstance(item, dict) and str(item.get("message", "")).strip()
    ]
    normalized_dropped_annotations = _normalize_dropped_annotations(dropped_annotations)
    return {
        "run_id": payload.get("run_id"),
        "warning_count": len(warnings),
        "warning_messages": warning_messages,
        "warnings": warnings,
        "files_with_infractions_count": len(_build_file_infractions(normalized_dropped_annotations)),
        "file_infractions": _build_file_infractions(normalized_dropped_annotations),
    }


def write_run_warnings(
    path: Path,
    report: RunReportModel | dict[str, Any],
    *,
    dropped_annotations: list[DroppedAnnotationModel] | list[dict[str, Any]] | None = None,
) -> Path | None:
    payload = build_run_warnings_payload(report, dropped_annotations=dropped_annotations)
    if int(payload.get("warning_count", 0)) <= 0:
        return None
    atomic_write_json(
        path,
        payload,
    )
    return path


def write_dropped_annotations(
    path: Path,
    *,
    run_id: str,
    dropped_annotations: list[DroppedAnnotationModel] | list[dict[str, Any]] | None = None,
) -> Path | None:
    normalized = _normalize_dropped_annotations(dropped_annotations)
    if not normalized:
        return None

    atomic_write_json(
        path,
        {
            "run_id": run_id,
            "dropped_annotation_count": len(normalized),
            "dropped_annotations": normalized,
        },
    )
    return path


def persist_run_artifacts(
    artifacts_dir: Path,
    run_id: str,
    config: RunConfigModel | dict[str, Any],
    report: RunReportModel | dict[str, Any],
    *,
    dropped_annotations: list[DroppedAnnotationModel] | list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    ensure_directory(artifacts_dir)
    config_path = artifacts_dir / f"{run_id}.config.json"
    report_path = artifacts_dir / f"{run_id}.report.json"
    warnings_path = artifacts_dir / f"{run_id}.warnings.json"
    dropped_annotations_path = artifacts_dir / f"{run_id}.dropped_annotations.json"
    write_run_config(config_path, config)
    write_run_report(report_path, report)
    write_run_warnings(warnings_path, report, dropped_annotations=dropped_annotations)
    write_dropped_annotations(
        dropped_annotations_path,
        run_id=run_id,
        dropped_annotations=dropped_annotations,
    )
    return config_path, report_path
