# Bounding-Box Annotation Converter
## Detailed Design Specification

Version: 0.2  
Date: 2026-03-03  
Status: Draft

## 1. Purpose

Define a production-quality tool that:
- Converts bounding-box annotations between common formats (COCO, YOLO, and extensible to others).
- Remaps integer class IDs from one taxonomy to another.
- Supports explicit dropping of one or more labels during remap.
- Provides full feature parity between CLI and GUI.
- Scans a directory and infers likely annotation format with confidence and diagnostics.

This spec emphasizes:
- Unit-test-first engineering.
- Human-readable code and outputs.
- Controlled, reviewable changes using Git.

## 2. Scope

In scope:
- 2D bounding boxes only.
- Batch conversion of datasets on local filesystem.
- Label remapping integrated into conversion pipeline or standalone.
- Format inference from directory content and sample files.
- Validation and dry-run reporting.
- CLI for all features and GUI as a usability layer over the same core services.
- Browser-accessible GUI mode for collaborative use by multiple users on a shared host/network.
- External dataset import from Kaggle, Roboflow, GitHub, and user-specified direct URLs.

Out of scope (v1):
- Segmentation masks, keypoints, polygons.
- Full hosted SaaS dataset management platform.
- Active learning workflows.
- Model training/inference.

## 3. Success Criteria

Functional:
- `convert` supports COCO JSON <-> YOLO TXT with deterministic outputs.
- `remap` supports one-to-one, many-to-one, and drop semantics.
- `infer` returns top format candidate plus confidence and evidence.
- CLI can execute all core actions without GUI.
- GUI can execute same core actions through shared APIs.
- External import supports Kaggle, Roboflow, GitHub, and arbitrary URL with provenance metadata.

Quality:
- Unit test coverage >= 90% on core domain + adapters.
- Round-trip conversion tests pass within tolerance bounds.
- No silent data loss: dropped, skipped, and invalid annotations are always reported.
- Strict lint + type checks enforced in CI.

## 4. User Personas and Key Use Cases

Dataset Engineer:
- Converts legacy labels from COCO to YOLO for training pipelines.
- Remaps old class taxonomy to new model class IDs.

Researcher:
- Uses GUI for quick inspection and ad hoc conversion.
- Uses CLI in scripts for reproducible experiments.

QA/ML Ops:
- Runs `infer` and `validate` before conversion.
- Reviews summary reports for dropped classes and malformed boxes.

## 5. Functional Requirements

### 5.1 Format Support (v1)

Required:
- COCO detection JSON.
- YOLO TXT + class file (`classes.txt` or explicit mapping config).

Planned extension points:
- Pascal VOC XML.
- Custom adapter plugins.

### 5.2 Conversion

Requirements:
- Preserve image-level identity (`file_name`, dimensions where available).
- Normalize internal representation to absolute pixel `xywh` for transformations.
- Handle coordinate conversions with explicit rounding policy.
- Support overwrite control and output directory management.

Modes:
- Full dataset conversion.
- Subset conversion (include/exclude patterns).
- Dry-run conversion with report only.

### 5.3 Label Remapping

Requirements:
- Accept mapping via CLI flags or mapping file (YAML/JSON/CSV).
- Support drop behavior by mapping to `null` (or `-1` when using numeric-only map file).
- Support many-to-one class merges.
- Require explicit policy for unmapped labels:
- `error` (default for safety)
- `drop`
- `identity`

Outputs:
- Detailed remap report:
- counts by source class
- counts by destination class
- dropped count and IDs
- unmapped count and IDs

### 5.4 Format Inference

Requirements:
- Scan directory recursively (configurable depth).
- Use deterministic scoring heuristics and return confidence.
- Provide evidence list and conflicting indicators.
- Fail closed on low confidence unless user uses `--force`.

Inference output:
- `predicted_format`
- `confidence` (0-1)
- `candidates` sorted by score
- `evidence` per candidate
- `warnings`

### 5.5 Validation

Checks:
- Annotation syntax and required fields.
- Bbox validity (`w > 0`, `h > 0`, within bounds unless policy allows clipping).
- Class ID existence in taxonomy.
- Image references exist and dimensions can be resolved.

Policies:
- `strict`: invalid annotation fails run.
- `permissive`: invalid annotation skipped with warnings.

### 5.6 CLI and GUI Parity

Requirement:
- Every GUI action maps to an underlying CLI-equivalent service operation.
- GUI must not implement separate conversion logic.

### 5.7 External Data Ingestion

Requirements:
- Support pulling datasets from Kaggle references.
- Support pulling datasets from Roboflow references.
- Support pulling datasets from GitHub repositories/releases.
- Support pulling datasets from direct user-provided URLs.
- Record provenance for every import:
- source type and source reference
- retrieval timestamp
- artifact checksum or integrity status
- Fail closed when import validation fails unless user explicitly overrides.

## 6. Non-Functional Requirements

- Language/runtime: Python 3.11+.
- Performance target: process 100k annotations in < 60s on typical workstation for COCO->YOLO (excluding image IO if dimensions already known).
- Deterministic outputs when input and config are identical.
- Logs are structured and human-readable.
- Errors include actionable remediation suggestions.
- Code style prioritizes readability over cleverness.

## 7. High-Level Architecture

Design principle: `Core library first`, interfaces second.

Layers:
- `core/domain`: canonical entities and business rules.
- `core/services`: conversion, remap, inference, validation orchestration.
- `adapters/formats`: COCO, YOLO readers/writers implementing shared interfaces.
- `interfaces/cli`: Typer-based command surface.
- `interfaces/gui`: desktop/web UI calling same services.
- `infra`: filesystem, logging, config loading.

Dependency direction:
- UI layers depend on services.
- Services depend on abstract adapter interfaces.
- Adapters depend on domain model.
- Domain has no outward dependency on UI.

## 8. Canonical Data Model

`Dataset`:
- `images: list[ImageRecord]`
- `annotations: list[Annotation]`
- `categories: dict[int, Category]`

`ImageRecord`:
- `image_id: str`
- `file_name: str`
- `width: int`
- `height: int`

`Annotation`:
- `annotation_id: str`
- `image_id: str`
- `class_id: int`
- `bbox_xywh_abs: tuple[float, float, float, float]`
- `iscrowd: bool | None`

`RemapConfig`:
- `class_map: dict[int, int | None]`
- `unmapped_policy: Literal["error", "drop", "identity"]`

Rationale:
- Canonical absolute `xywh` avoids repeated precision loss between normalized and pixel coordinates.

## 9. Adapter Contracts

`FormatReader` interface:
- `can_read(path) -> bool`
- `read(path, options) -> Dataset`

`FormatWriter` interface:
- `write(dataset, output_path, options) -> WriteReport`

`FormatDetector` interface:
- `score(path, sample_limit) -> DetectionScore`

Contract requirements:
- Readers must validate schema and return typed errors.
- Writers must emit deterministic ordering.
- Detectors must provide evidence strings for scores.

## 10. Format Inference Design

Scoring approach:
- Each format detector returns `score` and `evidence`.
- Global inference selects highest score if:
- top score >= `min_confidence`
- top score - second score >= `min_margin`

Example heuristics:

COCO indicators:
- Presence of JSON with top-level keys `images`, `annotations`, `categories`.
- `annotations[*].bbox` with 4-element arrays.

YOLO indicators:
- Parallel image and `.txt` label files.
- Label row structure: `class cx cy w h` with numeric tokens.
- Optional `classes.txt` or dataset yaml conventions.

Ambiguity handling:
- If confidence below threshold, return `ambiguous` with candidates.
- CLI/GUI asks user to pick format or force with explicit flag.

## 11. Label Mapping Semantics

Input map examples:

YAML:
```yaml
class_map:
  0: 0
  1: 4
  2: null   # drop
  3: 4      # merge class 3 into class 4
unmapped_policy: error
```

CSV:
```csv
src_class,dst_class
0,0
1,4
2,
3,4
```

Rules:
- If `dst_class` is null/empty => drop annotation.
- If source class missing in map, apply `unmapped_policy`.
- If destination class not in output taxonomy and taxonomy is fixed, error.
- If taxonomy is generated, create destination classes from observed mapped IDs.

## 12. CLI Specification

Executable name: `annobox`

Commands:
- `annobox infer --input <path> [--json] [--sample-limit N]`
- `annobox validate --input <path> --format <auto|coco|yolo> [--strict|--permissive]`
- `annobox convert --input <path> --output <path> --src <auto|coco|yolo> --dst <coco|yolo> [--map <file>] [--unmapped-policy ...] [--dry-run]`
- `annobox remap --input <path> --output <path> --format <coco|yolo> --map <file> [--dry-run]`
- `annobox report --input <conversion-report.json>`

Global flags:
- `--log-level <debug|info|warn|error>`
- `--log-file <path>`
- `--config <path>`
- `--seed <int>` for deterministic sampling when inference samples files.

Exit codes:
- `0` success
- `2` validation failure
- `3` format inference ambiguous/failed
- `4` conversion runtime error
- `5` config/mapping error

## 13. GUI Specification

Recommended implementation: web-based GUI for collaboration (Streamlit or equivalent), backed by
the same service layer used by CLI.

Primary screens:
- Home: choose input/output paths and task.
- Imports: fetch dataset from Kaggle/Roboflow/GitHub/direct URL and view provenance.
- Inference: scan directory, view confidence/evidence, confirm format.
- Mapping Editor: tabular src->dst mapping, drop toggles, unmapped policy.
- Conversion Run: start/cancel, live logs, summary stats.
- Report Viewer: dropped labels, invalid rows, output paths.

UX requirements:
- Never run destructive writes without explicit confirmation.
- Support multiple users viewing and launching runs from a shared web deployment.
- Show predicted impact before run:
- total files
- total annotations
- labels to drop
- classes merged
- validation errors expected
- Save/load mapping presets.
- Allow export of run configuration so results are reproducible via CLI.

## 14. Configuration

Single config file format: YAML.

Example:
```yaml
input: ./data/raw_dataset
output: ./data/converted_dataset
src_format: auto
dst_format: yolo
mapping_file: ./maps/classes_v2.yaml
unmapped_policy: error
validation_mode: strict
inference:
  min_confidence: 0.7
  min_margin: 0.15
  sample_limit: 100
imports:
  provider: kaggle
  source_ref: owner/dataset-name
  allow_unsafe: false
```

Precedence:
- CLI flags override config file.
- Config file overrides defaults.

## 15. Logging and Reporting

Runtime logging:
- Console human-readable logs.
- Optional JSONL machine-readable logs.

Run report (`conversion-report.json`):
- tool version and git commit hash (if available)
- run timestamp
- input/output paths
- source/destination format
- mapping summary
- validation errors
- per-class counts before/after
- dropped/unmapped details
- import provenance metadata (provider, source reference, integrity results)

## 16. Error Handling Policy

Principles:
- Fail early for config/schema errors.
- Fail safe for ambiguous inference by default.
- Never silently coerce invalid boxes without report.

Typed error classes:
- `ConfigError`
- `InferenceError`
- `ValidationError`
- `ConversionError`
- `MappingError`

## 17. Testing Strategy

### 17.1 Test Pyramid

Unit tests (majority):
- Domain model validation.
- Coordinate transform utilities.
- Mapping logic and policies.
- Inference heuristics with synthetic fixtures.
- Reader/writer adapter behavior in isolation.

Integration tests:
- End-to-end convert flows on fixture datasets.
- CLI command invocation and exit codes.
- GUI service calls (headless where practical).

Property tests:
- Round-trip invariants where expected:
- COCO -> canonical -> COCO preserves key fields.
- YOLO normalize/denormalize remains within tolerance.

Regression tests:
- Every bug fix adds a fixture + test.

### 17.2 Coverage and Quality Gates

Mandatory gates in CI:
- `ruff check`
- `black --check` or equivalent formatter
- `mypy`
- `pytest -q --maxfail=1`
- coverage threshold >= 90% on `core` and `adapters`

### 17.3 Fixture Design

Create compact fixtures for:
- valid COCO with multiple classes
- valid YOLO with classes file
- malformed rows and out-of-bounds boxes
- mixed directory content for inference ambiguity
- remap with drops and many-to-one merges
- imported dataset fixtures for Kaggle/Roboflow/GitHub/direct URL flows

## 18. Git Workflow and Controlled Changes

Branching:
- short-lived feature branches from `main`.
- branch naming: `feat/<topic>`, `fix/<topic>`, `test/<topic>`.

Commits:
- small, atomic, single-purpose.
- include tests in same commit when possible.
- message style: Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`).

Pull requests:
- required checklist:
- tests added/updated
- backward compatibility noted
- migration notes if output format changed
- sample report attached for conversion-affecting changes

Protection:
- require CI pass before merge.
- require at least one review.
- disallow force-push on protected branches.

Change-risk controls:
- Use feature flags for high-impact behavior changes.
- Keep adapter interfaces stable and versioned.
- Preserve old behavior behind explicit config if changing defaults.

## 19. Proposed Repository Structure

```text
annobox/
  pyproject.toml
  src/annobox/
    core/
      domain.py
      mapping.py
      bbox_math.py
      services/
        convert_service.py
        infer_service.py
        validate_service.py
    adapters/
      coco/
        reader.py
        writer.py
        detector.py
      yolo/
        reader.py
        writer.py
        detector.py
    interfaces/
      cli/
        main.py
      gui/
        app.py
        viewmodels.py
    infra/
      fs.py
      logging.py
      config.py
    reports/
      models.py
      writer.py
  tests/
    unit/
    integration/
    fixtures/
```

## 20. Security and Safety Considerations

- Reject path traversal attempts in file discovery/output paths.
- Treat input files as untrusted; robust parsing with clear exceptions.
- Limit recursion and sample sizes to avoid accidental resource exhaustion.
- GUI should display exact output path before run to prevent accidental overwrite.
- Protect provider credentials/tokens and avoid writing secrets to logs or reports.
- Validate downloaded artifacts before extraction and conversion.

## 21. Performance Considerations

- Stream large JSON where possible.
- Avoid loading image binaries unless dimensions are missing.
- Parallelize per-file operations only when deterministic ordering/reporting can be preserved.
- Cache inferred image dimensions by file hash/path + mtime.

## 22. Milestones

M1: Core domain + COCO/YOLO read/write + remap engine + unit tests  
M2: Inference engine + validation + CLI commands + integration tests  
M3: External import providers + provenance tracking + import validation  
M4: Web GUI workflows + collaborative run management + report viewer + UX polish  
M5: Hardening: CI quality gates, performance tuning, docs, release candidate

## 23. Acceptance Test Checklist

- Convert COCO->YOLO on fixture dataset, verify class counts and bbox tolerance.
- Convert YOLO->COCO with stable deterministic output ordering.
- Remap with drop and many-to-one, verify report counts.
- Inference correctly identifies clean COCO and clean YOLO datasets.
- Inference returns ambiguous on mixed-format directory and blocks unless forced.
- CLI and GUI produce equivalent conversion report for same config.
- Imports from Kaggle/Roboflow/GitHub/direct URL produce validated local dataset + provenance info.

## 24. Open Questions

- Should Pascal VOC support be included in v1 or v1.1?
- Should output taxonomy always be explicit or inferred when remapping?
- What tolerance should be default for float rounding in round-trip assertions?
- Should Streamlit be the default web GUI framework, or should an equivalent be selected?

## 25. Recommended Defaults (v1)

- `unmapped_policy = error`
- `validation_mode = strict`
- `inference.min_confidence = 0.70`
- `inference.min_margin = 0.15`
- deterministic writer ordering enabled
- report writing always on (can be directed to temp path)

## 26. Final Implementation Notes (2026-03-04)

Implemented v1 behavior in `src/label_master` with these concretions:

- Canonical domain model and typed policies/errors implemented in `core/domain`.
- Deterministic COCO/YOLO detect-read-write adapters implemented in `adapters/coco` and `adapters/yolo`.
- Shared orchestration services implemented for infer/validate/remap/convert/import in `core/services`.
- Dry-run known-bbox manifest verification implemented with YAML manifests and zero-write conversion mode.
- Typer CLI (`annobox`) implemented with infer/validate/convert/remap/import flows and run artifact persistence.
- Streamlit GUI implemented as localhost-only workflow using shared services and portable run-config export.
- Provider imports implemented for Kaggle/Roboflow/GitHub (local path adapters for v1 test fixtures) and direct URL (`https|http|file`) with protocol warnings.
- Output-path contention manager implemented with last-write-wins metadata and report event integration.
- CI policy scripts and workflow implemented for lint/type/test/coverage + traceability and regression evidence gates.
- Contract/integration/unit coverage expanded with benchmarks, compatibility tests (v1 current + N-1), and regression-policy gates.
