from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceFormat(str, Enum):
    AUTO = "auto"
    COCO = "coco"
    YOLO = "yolo"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class RunMode(str, Enum):
    INFER = "infer"
    VALIDATE = "validate"
    CONVERT = "convert"
    REMAP = "remap"
    IMPORT = "import"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ContentionResolution(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"


class ImportProvider(str, Enum):
    KAGGLE = "kaggle"
    ROBOFLOW = "roboflow"
    GITHUB = "github"
    DIRECT_URL = "direct_url"


class ImportProtocol(str, Enum):
    HTTPS = "https"
    HTTP = "http"
    FILE = "file"


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    checksum: str | None = None


class AnnotationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    class_id: int
    bbox_xywh_abs: tuple[float, float, float, float]
    iscrowd: bool | None = None
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("bbox_xywh_abs")
    @classmethod
    def _validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, w, h = value
        if w <= 0 or h <= 0:
            raise ValueError("bbox width/height must be positive")
        return value


class CategoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int
    name: str = Field(min_length=1)
    supercategory: str | None = None


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: str = Field(min_length=1)
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    loader: str = Field(min_length=1)


class AnnotationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    source_format: SourceFormat
    images: list[ImageRecord]
    annotations: list[AnnotationRecord]
    categories: dict[int, CategoryRecord]
    source_metadata: SourceMetadata

    @model_validator(mode="after")
    def _validate_links(self) -> "AnnotationDataset":
        image_ids = {img.image_id for img in self.images}
        if len(image_ids) != len(self.images):
            raise ValueError("duplicate image_id entries are not allowed")
        for annotation in self.annotations:
            if annotation.image_id not in image_ids:
                raise ValueError(
                    f"annotation {annotation.annotation_id} references missing image {annotation.image_id}"
                )
            if annotation.class_id not in self.categories:
                raise ValueError(
                    f"annotation {annotation.annotation_id} references missing class {annotation.class_id}"
                )
        return self

    def deterministic_annotations(self) -> list[AnnotationRecord]:
        return sorted(
            self.annotations,
            key=lambda a: (a.image_id, a.class_id, a.annotation_id),
        )


class WarningEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Severity
    context: dict[str, str] = Field(default_factory=dict)


class ContentionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    competing_run_id: str = Field(min_length=1)
    resolution: ContentionResolution = ContentionResolution.LAST_WRITE_WINS
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InferenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: SourceFormat
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_format: SourceFormat
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: list[InferenceCandidate]
    warnings: list[WarningEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_sorted_candidates(self) -> "InferenceResult":
        scores = [candidate.score for candidate in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("candidates must be sorted by descending score")
        return self


class SummaryCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: int = Field(ge=0)
    annotations_in: int = Field(ge=0)
    annotations_out: int = Field(ge=0)
    dropped: int = Field(ge=0)
    unmapped: int = Field(ge=0)
    invalid: int = Field(ge=0)
    skipped: int = Field(ge=0)


class MappingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_map_size: int = Field(default=0, ge=0)
    dropped_class_ids: list[int] = Field(default_factory=list)
    unmapped_class_ids: list[int] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    invalid_annotations: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class ImportSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: ImportProvider
    reference: str = Field(min_length=1)
    direct_url_protocol: ImportProtocol | None = None
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ImportArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    source: ImportSource
    local_path: str = Field(min_length=1)
    integrity_status: str = Field(pattern="^(passed|failed)$")
    validation_status: str = Field(pattern="^(passed|failed)$")
    warnings: list[WarningEvent] = Field(default_factory=list)


class ImportProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ImportProvider
    source_ref: str = Field(min_length=1)
    protocol: ImportProtocol | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    integrity_status: str = Field(pattern="^(passed|failed)$")
    checksum_status: str | None = Field(default=None, pattern="^(passed|failed|unknown)$")
    import_job_id: str = Field(min_length=1)


class ConversionRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: RunMode
    input_path: str = Field(min_length=1)
    output_path: str | None = None
    src_format: SourceFormat
    dst_format: SourceFormat | None = None
    status: RunStatus = RunStatus.CREATED
    warnings: list[WarningEvent] = Field(default_factory=list)
    contention_events: list[ContentionEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_paths(self) -> "ConversionRun":
        if self.mode in {RunMode.CONVERT, RunMode.REMAP, RunMode.IMPORT} and not self.output_path:
            raise ValueError("output_path is required for convert/remap/import modes")
        return self


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = Field(pattern="^(completed|failed)$")
    tool_version: str = Field(default="0.1.0", min_length=1)
    git_commit: str | None = None
    input_path: str = Field(min_length=1)
    output_path: str | None = None
    src_format: SourceFormat | None = None
    dst_format: SourceFormat | None = None
    summary_counts: SummaryCounts
    mapping_summary: MappingSummary = Field(default_factory=MappingSummary)
    validation_summary: ValidationSummary = Field(default_factory=lambda: ValidationSummary(valid=True))
    warnings: list[WarningEvent] = Field(default_factory=list)
    contention_events: list[ContentionEvent] = Field(default_factory=list)
    provenance: list[ImportProvenance] = Field(default_factory=list)
