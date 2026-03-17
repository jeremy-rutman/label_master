from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from label_master.adapters.providers.direct_url_provider import fetch_direct_url
from label_master.adapters.providers.github_provider import fetch_github_dataset
from label_master.adapters.providers.kaggle_provider import fetch_kaggle_dataset
from label_master.adapters.providers.roboflow_provider import fetch_roboflow_dataset
from label_master.core.domain.value_objects import ImportError
from label_master.reports.schemas import (
    ProvenanceModel,
    RunReportModel,
    SummaryCountsModel,
    WarningEventModel,
)


@dataclass(frozen=True)
class ImportResult:
    local_path: Path
    report: RunReportModel


def _has_annotation_payload(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in {".zip", ".json", ".txt", ".yaml", ".yml"}

    files = list(path.rglob("*"))
    return any(file.is_file() and file.suffix.lower() in {".json", ".txt", ".yaml", ".yml"} for file in files)


def import_dataset(
    *,
    provider: str,
    source_ref: str,
    output_path: Path,
    run_id: str,
) -> ImportResult:
    provider_name = provider.lower()

    if provider_name == "kaggle":
        fetched = fetch_kaggle_dataset(source_ref, output_path)
    elif provider_name == "roboflow":
        fetched = fetch_roboflow_dataset(source_ref, output_path)
    elif provider_name == "github":
        fetched = fetch_github_dataset(source_ref, output_path)
    elif provider_name == "direct_url":
        fetched = fetch_direct_url(
            source_ref,
            output_path,
            allowed_file_root=output_path.parent.resolve(),
        )
    else:
        raise ImportError(f"Unsupported provider: {provider}")

    if not fetched.local_path.exists():
        raise ImportError("Imported artifact path does not exist after provider fetch")

    if not _has_annotation_payload(fetched.local_path):
        raise ImportError("Imported artifact failed schema/integrity gating: no annotation-like files found")

    warning_models = [
        WarningEventModel(
            code="import_protocol_warning",
            message=message,
            severity="warning",
            context={"provider": provider_name, "source_ref": source_ref},
        )
        for message in (fetched.warnings or [])
    ]

    report = RunReportModel(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        status="completed",
        input_path=source_ref,
        output_path=str(fetched.local_path),
        summary_counts=SummaryCountsModel(
            images=0,
            annotations_in=0,
            annotations_out=0,
            dropped=0,
            unmapped=0,
            invalid=0,
            skipped=0,
        ),
        warnings=warning_models,
        provenance=[
            ProvenanceModel(
                provider=provider_name,  # type: ignore[arg-type]
                source_ref=source_ref,
                protocol=fetched.protocol,  # type: ignore[arg-type]
                retrieved_at=datetime.now(UTC),
                integrity_status="passed",
                checksum_status="unknown",
                import_job_id=run_id,
            )
        ],
    )

    return ImportResult(local_path=fetched.local_path, report=report)
