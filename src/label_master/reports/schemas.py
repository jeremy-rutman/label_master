from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from label_master.core.domain.value_objects import ConfigurationError

CURRENT_V1_MINOR = 1
PREVIOUS_V1_MINOR = 0
CURRENT_SCHEMA_VERSION = f"1.{CURRENT_V1_MINOR}"
PREVIOUS_SCHEMA_VERSION = f"1.{PREVIOUS_V1_MINOR}"
SUPPORTED_SCHEMA_VERSIONS = {CURRENT_SCHEMA_VERSION, PREVIOUS_SCHEMA_VERSION}


class WarningEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"]
    context: dict[str, str] = Field(default_factory=dict)


class ContentionEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    competing_run_id: str = Field(min_length=1)
    resolution: Literal["last_write_wins"] = "last_write_wins"
    resolved_at: datetime | None = None


class ProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["kaggle", "roboflow", "github", "direct_url"]
    source_ref: str = Field(min_length=1)
    protocol: Literal["https", "http", "file"] | None = None
    retrieved_at: datetime
    integrity_status: Literal["passed", "failed"]
    checksum_status: Literal["passed", "failed", "unknown"] | None = None
    import_job_id: str = Field(min_length=1)


class SummaryCountsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: int = Field(ge=0)
    annotations_in: int = Field(ge=0)
    annotations_out: int = Field(ge=0)
    dropped: int = Field(ge=0)
    unmapped: int = Field(ge=0)
    invalid: int = Field(ge=0)
    skipped: int = Field(ge=0)


class RunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    mode: Literal["infer", "validate", "convert", "remap", "import"]
    input_path: str = Field(min_length=1)
    output_path: str | None = None
    src_format: Literal["auto", "coco", "yolo"]
    dst_format: Literal["coco", "yolo"] | None = None
    mapping_file: str | None = None
    unmapped_policy: Literal["error", "drop", "identity"] = "error"
    dry_run: bool = False
    provider: Literal["kaggle", "roboflow", "github", "direct_url"] | None = None
    source_ref: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> "RunConfigModel":
        if self.mode == "convert":
            if not self.output_path or not self.dst_format:
                raise ValueError("convert mode requires output_path and dst_format")
        if self.mode == "remap":
            if not self.output_path or not self.mapping_file:
                raise ValueError("remap mode requires output_path and mapping_file")
        if self.mode == "import":
            if not self.output_path or not self.provider or not self.source_ref:
                raise ValueError("import mode requires output_path, provider, and source_ref")
        return self


class RunReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    timestamp: datetime
    status: Literal["completed", "failed"]
    tool_version: str = Field(default="0.1.0")
    git_commit: str | None = None
    input_path: str = Field(min_length=1)
    output_path: str | None = None
    src_format: Literal["auto", "coco", "yolo"] | None = None
    dst_format: Literal["coco", "yolo"] | None = None
    summary_counts: SummaryCountsModel
    warnings: list[WarningEventModel] = Field(default_factory=list)
    contention_events: list[ContentionEventModel] = Field(default_factory=list)
    provenance: list[ProvenanceModel] = Field(default_factory=list)


def negotiate_schema_version(requested: str | None) -> str:
    if requested is None:
        return CURRENT_SCHEMA_VERSION
    if requested not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigurationError(
            f"Unsupported artifact schema version: {requested}",
            context={"supported": ",".join(sorted(SUPPORTED_SCHEMA_VERSIONS))},
        )
    return requested


def _extract_version(payload: dict[str, Any]) -> str:
    raw_version = payload.get("schema_version")
    if raw_version is None:
        return PREVIOUS_SCHEMA_VERSION
    if not isinstance(raw_version, str):
        raise ConfigurationError("schema_version must be a string")
    return negotiate_schema_version(raw_version)


def upgrade_run_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = _extract_version(payload)
    upgraded = dict(payload)

    if version == PREVIOUS_SCHEMA_VERSION:
        upgraded.setdefault("dry_run", False)
        upgraded.setdefault("unmapped_policy", "error")
        upgraded.setdefault("provider", None)
        upgraded.setdefault("source_ref", None)

    upgraded["schema_version"] = CURRENT_SCHEMA_VERSION
    return upgraded


def upgrade_run_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = _extract_version(payload)
    upgraded = dict(payload)

    if version == PREVIOUS_SCHEMA_VERSION:
        upgraded.setdefault("tool_version", "0.1.0")
        provenance = upgraded.get("provenance")
        if isinstance(provenance, list):
            for entry in provenance:
                if isinstance(entry, dict):
                    entry.setdefault("checksum_status", None)

    upgraded["schema_version"] = CURRENT_SCHEMA_VERSION
    return upgraded


def parse_run_config(payload: dict[str, Any]) -> RunConfigModel:
    return RunConfigModel.model_validate(upgrade_run_config_payload(payload))


def parse_run_report(payload: dict[str, Any]) -> RunReportModel:
    return RunReportModel.model_validate(upgrade_run_report_payload(payload))


def load_run_config(path: Path) -> RunConfigModel:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigurationError("Run config payload must be an object")
    return parse_run_config(payload)


def load_run_report(path: Path) -> RunReportModel:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigurationError("Run report payload must be an object")
    return parse_run_report(payload)
