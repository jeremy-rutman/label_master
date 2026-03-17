# CLI Contract

**Feature**: `001-annotation-collab-ingestion`  
**Interface Type**: User-facing command contract

## Command Surface

Executable: `annobox`

### 1) Infer

```bash
annobox infer --input <path> [--json] [--sample-limit <int>] [--force]
```

- **Behavior**:
  - Scans input path for known annotation indicators.
  - Returns predicted format, confidence, candidates, and evidence.
  - Fails with exit code `3` when ambiguous and `--force` is not provided.

### 2) Validate

```bash
annobox validate --input <path> --format <auto|coco|yolo> [--strict|--permissive]
```

- **Behavior**:
  - Validates schema, bbox constraints, image/category references.
  - `--strict` fails on invalid records.
  - `--permissive` allows invalid-record skipping with warning counts.

### 3) Convert

```bash
annobox convert \
  --input <path> \
  --output <path> \
  --src <auto|coco|yolo> \
  --dst <coco|yolo> \
  [--map <path>] \
  [--unmapped-policy <error|drop|identity>] \
  [--dry-run]
```

- **Behavior**:
  - Performs infer/validate/remap/convert through shared core services.
  - Emits deterministic outputs and JSON report.
  - When multiple runs target same output path concurrently, last completed run owns final files;
    contention must be recorded in both reports.

### 4) Remap

```bash
annobox remap --input <path> --output <path> --format <coco|yolo> --map <path> [--dry-run]
```

- **Behavior**:
  - Applies class mapping and drop semantics without changing format.

### 5) Import

```bash
annobox import \
  --provider <kaggle|roboflow|github|direct_url> \
  --source-ref <value> \
  --output <path>
```

- **Behavior**:
  - Downloads/imports dataset into local workspace.
  - Allows direct URL protocols `https://`, `http://`, `file://`.
  - Emits warnings for `http://` and `file://` before retrieval continues.
  - Fails closed when integrity/schema validation fails.

## Global Options

```text
--config <path>
--log-level <debug|info|warn|error>
--log-file <path>
--report-path <path>
```

## Exit Codes

- `0`: Success
- `2`: Validation failure
- `3`: Inference ambiguous/failure without override
- `4`: Conversion/runtime error
- `5`: Configuration/mapping/import reference error

## Output Artifacts

- Run config document MUST validate against `run-config.schema.json`.
- Run report document MUST validate against `run-report.schema.json`.

## GUI Parity Rule

Any GUI-triggered operation MUST map to one equivalent CLI invocation with matching parameters and
equivalent output/report behavior.
