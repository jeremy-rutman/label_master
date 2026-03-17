from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from label_master.core.domain.value_objects import ConfigurationError
from label_master.reports.schemas import (
    RunConfigModel,
    RunReportModel,
    parse_run_config,
    parse_run_report,
    upgrade_run_config_payload,
    upgrade_run_report_payload,
)


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file must contain an object/map: {path}")
    return {str(k): v for k, v in data.items()}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(
    *,
    defaults: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    if defaults:
        resolved = _deep_merge(resolved, defaults)
    if config_path:
        resolved = _deep_merge(resolved, _load_config_file(config_path))
    if overrides:
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
        resolved = _deep_merge(resolved, filtered_overrides)
    return resolved


def load_run_config_model(path: Path) -> RunConfigModel:
    payload = _load_config_file(path)
    upgraded = upgrade_run_config_payload(payload)
    return parse_run_config(upgraded)


def load_run_report_model(path: Path) -> RunReportModel:
    payload = _load_config_file(path)
    upgraded = upgrade_run_report_payload(payload)
    return parse_run_report(upgraded)


def load_mapping_file(path: Path) -> dict[int, int | None]:
    raw = _load_config_file(path)
    class_map_raw = raw.get("class_map", raw)
    if not isinstance(class_map_raw, dict):
        raise ConfigurationError("Mapping file must contain a dictionary")

    parsed: dict[int, int | None] = {}
    for key, value in class_map_raw.items():
        try:
            source_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid source class id key: {key!r}") from exc

        if value is None:
            parsed[source_id] = None
            continue

        try:
            parsed[source_id] = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid destination class id value: {value!r}") from exc

    return parsed
