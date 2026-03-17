from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from label_master.infra.filesystem import atomic_write_json, ensure_directory
from label_master.reports.schemas import RunConfigModel, RunReportModel


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


def persist_run_artifacts(
    artifacts_dir: Path,
    run_id: str,
    config: RunConfigModel | dict[str, Any],
    report: RunReportModel | dict[str, Any],
) -> tuple[Path, Path]:
    ensure_directory(artifacts_dir)
    config_path = artifacts_dir / f"{run_id}.config.json"
    report_path = artifacts_dir / f"{run_id}.report.json"
    write_run_config(config_path, config)
    write_run_report(report_path, report)
    return config_path, report_path
