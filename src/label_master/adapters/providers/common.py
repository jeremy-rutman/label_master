from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from label_master.core.domain.value_objects import ImportError


@dataclass(frozen=True)
class ProviderFetchResult:
    local_path: Path
    protocol: str | None = None
    warnings: list[str] | None = None


def copy_into_output(source_path: Path, output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        destination = output_path / source_path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        return destination

    destination = output_path / source_path.name
    shutil.copy2(source_path, destination)
    return destination


def parse_scheme(source_ref: str) -> str | None:
    parsed = urlparse(source_ref)
    return parsed.scheme or None


def require_existing_local_path(source_ref: str) -> Path:
    source_path = Path(source_ref)
    if not source_path.exists():
        raise ImportError(f"Source path does not exist: {source_ref}")
    return source_path
