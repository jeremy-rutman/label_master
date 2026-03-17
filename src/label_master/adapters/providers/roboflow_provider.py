from __future__ import annotations

from pathlib import Path

from label_master.adapters.providers.common import (
    ProviderFetchResult,
    copy_into_output,
    require_existing_local_path,
)


def fetch_roboflow_dataset(source_ref: str, output_path: Path) -> ProviderFetchResult:
    source_path = require_existing_local_path(source_ref)
    local_path = copy_into_output(source_path, output_path)
    return ProviderFetchResult(local_path=local_path, protocol=None, warnings=[])
