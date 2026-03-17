# Data Model: Bounding-Box Conversion and Single-User Web GUI Ingestion

**Feature**: `001-annotation-collab-ingestion`  
**Date**: 2026-03-03

## 1. AnnotationDataset

Represents a dataset loaded into canonical form for validation, remapping, and conversion.

### Fields

- `dataset_id: str`
- `source_format: Literal["coco", "yolo", "unknown"]`
- `images: list[ImageRecord]`
- `annotations: list[AnnotationRecord]`
- `categories: dict[int, CategoryRecord]`
- `source_metadata: SourceMetadata`

### Validation Rules

- `dataset_id` MUST be non-empty and stable per run.
- `images` and `categories` MUST contain unique identities.
- `source_format` MUST be resolved before conversion writes.

## 2. ImageRecord

### Fields

- `image_id: str`
- `file_name: str`
- `width: int`
- `height: int`
- `checksum: str | None`

### Validation Rules

- `width > 0`, `height > 0`
- `file_name` MUST be relative to dataset root for portability.

## 3. AnnotationRecord

### Fields

- `annotation_id: str`
- `image_id: str`
- `class_id: int`
- `bbox_xywh_abs: tuple[float, float, float, float]`
- `iscrowd: bool | None`
- `attributes: dict[str, str | int | float | bool | None]`

### Validation Rules

- `image_id` MUST reference an existing `ImageRecord`.
- `bbox_xywh_abs = (x, y, w, h)` with `w > 0` and `h > 0`.
- Bounding boxes SHOULD be within image bounds unless clipping policy is explicitly enabled.

## 4. CategoryRecord

### Fields

- `class_id: int`
- `name: str`
- `supercategory: str | None`

### Validation Rules

- `class_id` MUST be unique in dataset taxonomy.
- `name` MUST be non-empty.

## 5. RemapConfig

### Fields

- `class_map: dict[int, int | None]`
- `unmapped_policy: Literal["error", "drop", "identity"]`

### Validation Rules

- Keys in `class_map` MUST be integers.
- Values MUST be integer destination class or `null` for drop.
- `unmapped_policy` defaults to `error`.

## 6. ImportSource

### Fields

- `source_type: Literal["kaggle", "roboflow", "github", "direct_url"]`
- `reference: str`
- `direct_url_protocol: Literal["https", "http", "file"] | None`
- `requested_at: datetime`

### Validation Rules

- `source_type` determines provider-specific reference validation.
- For `direct_url`, protocol MUST be one of `https`, `http`, `file`.
- `http` and `file` imports MUST produce warning events before retrieval continues.

## 7. ImportArtifact

### Fields

- `artifact_id: str`
- `source: ImportSource`
- `local_path: str`
- `integrity_status: Literal["passed", "failed"]`
- `validation_status: Literal["passed", "failed"]`
- `warnings: list[WarningEvent]`

### Validation Rules

- Conversion cannot proceed if integrity/validation status is `failed` unless explicit override.
- `local_path` MUST pass path traversal safety checks.

## 8. InferenceResult

### Fields

- `predicted_format: Literal["coco", "yolo", "ambiguous", "unknown"]`
- `confidence: float`
- `candidates: list[InferenceCandidate]`
- `warnings: list[WarningEvent]`

### Validation Rules

- `0.0 <= confidence <= 1.0`
- `candidates` MUST be sorted by descending score.
- Conversion MUST block when predicted format is `ambiguous` unless explicit override.

## 9. ConversionRun

### Fields

- `run_id: str`
- `created_at: datetime`
- `mode: Literal["infer", "validate", "convert", "remap", "import"]`
- `input_path: str`
- `output_path: str | None`
- `src_format: Literal["auto", "coco", "yolo"]`
- `dst_format: Literal["coco", "yolo"] | None`
- `status: Literal["created", "running", "completed", "failed"]`
- `warnings: list[WarningEvent]`
- `contention_events: list[ContentionEvent]`

### Validation Rules

- `run_id` MUST be unique and audit-friendly.
- For conversion/remap modes, `output_path` MUST be provided.
- For GUI-triggered runs, behavior MUST match equivalent CLI parameters.

## 10. RunReport

### Fields

- `run_id: str`
- `timestamp: datetime`
- `tool_version: str`
- `git_commit: str | null`
- `summary_counts: SummaryCounts`
- `mapping_summary: MappingSummary`
- `validation_summary: ValidationSummary`
- `provenance: list[ImportProvenance]`
- `warnings: list[WarningEvent]`
- `contention_events: list[ContentionEvent]`

### Validation Rules

- MUST include counts for dropped/skipped/invalid/unmapped records when non-zero.
- MUST include protocol warning records for `http`/`file` direct URL imports.
- MUST include contention details when same-output runs overlap.

## 11. Supporting Value Objects

### WarningEvent

- `code: str`
- `message: str`
- `severity: Literal["info", "warning", "error"]`
- `context: dict[str, str]`

### ContentionEvent

- `output_path: str`
- `run_id: str`
- `competing_run_id: str`
- `resolution: Literal["last_write_wins"]`
- `resolved_at: datetime`

### SourceMetadata

- `dataset_root: str`
- `loaded_at: datetime`
- `loader: str`

## Relationships

- `AnnotationDataset.images (1) -> (N) AnnotationRecord.image_id`
- `AnnotationDataset.categories (1) -> (N) AnnotationRecord.class_id`
- `ConversionRun (1) -> (1) RunReport`
- `ImportSource (1) -> (N) ImportArtifact`
- `ConversionRun (1) -> (N) WarningEvent`
- `ConversionRun (0..N) -> (N) ContentionEvent`

## State Transitions

### ImportArtifact lifecycle

`requested -> downloaded -> integrity_checked -> validated -> ready | failed`

### ConversionRun lifecycle

`created -> running -> completed | failed`

### InferenceResult lifecycle

`scored -> resolved(coco|yolo) | ambiguous | unknown`
