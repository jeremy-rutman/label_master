# Data Model: Streamlit GUI Workflow and Mapping Reliability Enhancements

**Feature**: `002-enhance-gui`  
**Date**: 2026-03-05

## 1. GuiDirectorySelectionState

Session-scoped state for input directory selection and validation feedback.

### Fields

- `input_dir_raw: str`
- `input_dir_resolved: str | None`
- `browse_available: bool`
- `browse_message: str | None`
- `validation_errors: list[str]`

### Validation Rules

- `input_dir_raw` MAY be empty only before user selection.
- `input_dir_resolved` MUST be absolute when set.
- If `input_dir_resolved` exists, it MUST point to a directory.

## 2. DirectoryValidationResult

Normalized result of validating dataset directory candidate.

### Fields

- `exists: bool`
- `is_directory: bool`
- `is_readable: bool`
- `errors: list[str]`

### Validation Rules

- `errors` MUST be deterministic for identical path state.
- Run execution MUST be blocked when `errors` is non-empty.

## 3. GuiMappingRow

One row entered in the mapping editor.

### Fields

- `source_class_id: str`
- `action: Literal["map", "drop"]`
- `destination_class_id: str`

### Validation Rules

- `source_class_id` MUST parse as integer unless row is empty.
- `destination_class_id` MUST parse as integer when `action == "map"`.
- Duplicate `source_class_id` rows are invalid.
- `destination_class_id` MAY be empty only when `action == "drop"`.

## 4. GuiMappingParseResult

Parse output produced from mapping rows.

### Fields

- `class_map: dict[int, int | None]`
- `errors: list[str]`

### Validation Rules

- Keys in `class_map` MUST be unique integers.
- Values MUST be integer or `null` (`drop` semantics).
- Parse results MUST be stable for identical row input ordering/content.

## 5. GuiRunReviewState

Aggregated state used to gate and execute conversion from Review & Run.

### Fields

- `input_dir: str`
- `output_dir: str`
- `src_format: Literal["auto", "coco", "yolo"]`
- `dst_format: Literal["coco", "yolo"]`
- `unmapped_policy: Literal["error", "drop", "identity"]`
- `dry_run: bool`
- `mapping_parse: GuiMappingParseResult`
- `blocking_errors: list[str]`

### Validation Rules

- `blocking_errors` MUST include path validation + mapping parse errors.
- Run action MUST be disabled when `blocking_errors` is non-empty.

## 6. GuiRunSummaryView

Display model for post-run summary reporting.

### Fields

- `run_id: str`
- `status: Literal["completed", "failed"]`
- `images_processed: int`
- `annotations_converted: int`
- `warning_count: int`
- `error_count: int`
- `config_path: str | None`
- `mapping_path: str | None`

### Validation Rules

- Numeric counts MUST be non-negative.
- `run_id` MUST match the run report identifier.

## 7. OutputDirectoryAccessResult

Result of invoking output-directory open action.

### Fields

- `requested_path: str`
- `opened: bool`
- `message: str`

### Validation Rules

- `requested_path` MUST be shown to user regardless of `opened` result.
- Failure to auto-open MUST NOT erase run completion state.

## Relationships

- `GuiDirectorySelectionState (1) -> (1) DirectoryValidationResult`
- `GuiMappingRow (N) -> (1) GuiMappingParseResult`
- `GuiRunReviewState (1) -> (1) GuiMappingParseResult`
- `GuiRunReviewState (1) -> (1) GuiRunSummaryView` after execution
- `GuiRunSummaryView (0..1) -> (1) OutputDirectoryAccessResult` after user action

## State Transitions

### Directory Selection

`empty -> selected -> validated_valid | validated_invalid -> selected`

### Mapping Editor

`rows_editing -> rows_parsed_valid | rows_parsed_invalid -> rows_editing`

### Run Lifecycle

`review_ready -> running -> completed | failed`
