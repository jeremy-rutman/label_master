# Implementation Plan: Bounding-Box Conversion and Single-User Web GUI Ingestion

**Branch**: `001-annotation-collab-ingestion` | **Date**: 2026-03-03 | **Spec**: [/home/jeremy/data/drone_detect/label_master/specs/001-annotation-collab-ingestion/spec.md](/home/jeremy/data/drone_detect/label_master/specs/001-annotation-collab-ingestion/spec.md)
**Input**: Feature specification from `/specs/001-annotation-collab-ingestion/spec.md`
**Branch Naming Note**: Branch slug is retained from initial "collab" naming for traceability; current v1 scope remains single-user.

## Summary

Deliver a Python-first annotation pipeline that converts COCO/YOLO bounding-box datasets,
performs class remapping with drop semantics, infers source format, validates data, and imports
external datasets from Kaggle/Roboflow/GitHub/direct URLs. Expose identical behavior through a
CLI and a Streamlit web GUI bound to localhost (single-user v1). Preserve deterministic outputs,
fail closed by default, and emit audit-ready run reports including contention/warning events.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: Typer (CLI), Streamlit (GUI), Pydantic v2 (models/validation), PyYAML
(config/map parsing), httpx (HTTP downloads), kaggle (provider adapter), pytest + pytest-cov +
Hypothesis (tests), ruff + mypy (quality gates)  
**Storage**: Local filesystem only (COCO JSON, YOLO TXT, YAML maps/config, JSON reports, downloaded
archives/artifacts)  
**Testing**: pytest (unit/integration/contract), Hypothesis for bbox transform invariants,
regression fixtures per bug fix, dry-run validation suites using user-provided sample datasets with
known bbox manifests, and run-artifact compatibility tests for current/previous v1 minor versions  
**Target Platform**: Linux/macOS development workstations (Windows-compatible paths where feasible)  
**Project Type**: Single Python package with shared core library, CLI interface, and localhost web
GUI interface  
**Performance Goals**: COCO<->YOLO conversion of 100k annotations in <60s (excluding image decode);
format inference over 10k files in <30s with bounded sampling  
**Constraints**: 2D bbox only; GUI binds localhost only; no auth/roles in v1; fail-closed default;
no silent destructive changes; deterministic ordering; dry-run performs no converted-annotation
writes; `http://` and `file://` imports allowed but must warn and be recorded in provenance  
**Scale/Scope**: Single-user execution on one host, up to ~50k images / ~100k annotations per run,
future multi-user support preserved through portable run artifacts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Gate

- **Core Library and Canonical Model First**: PASS
  - Plan isolates business rules in `core/domain` and `core/services`; CLI and GUI are adapters.
- **CLI-Complete and GUI-Parity Interfaces**: PASS
  - Every GUI action maps to the same service operations and run config used by CLI.
- **Safety, Determinism, and Auditability**: PASS
  - Plan includes fail-closed defaults, deterministic ordering, and required run/provenance reports.
- **Test-Driven Quality Gates**: PASS
  - Plan requires unit/integration/contract/regression tests and CI quality gates.
- **Spec-Driven Delivery and Change Control**: PASS
  - Work is explicitly tied to US1-US4, FR-001..FR-021 (+ FR-003a/003b, FR-010a, FR-015a,
    FR-019a/019b), and SC-001..SC-011.

### Post-Design Gate (Phase 1 Re-check)

- **Core Library and Canonical Model First**: PASS
  - `data-model.md` keeps canonical entities independent from CLI/GUI formats.
- **CLI-Complete and GUI-Parity Interfaces**: PASS
  - `contracts/cli-contract.md` and `quickstart.md` define reproducible parity workflows.
- **Safety, Determinism, and Auditability**: PASS
  - `contracts/run-report.schema.json` includes warnings, contention, provenance, and run identity.
- **Test-Driven Quality Gates**: PASS
  - `quickstart.md` includes contract/integration/unit test paths and gate commands.
- **Spec-Driven Delivery and Change Control**: PASS
  - Generated artifacts remain scoped to clarified spec and preserve deferred multi-user boundary.

## Project Structure

### Documentation (this feature)

```text
specs/001-annotation-collab-ingestion/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli-contract.md
│   ├── run-config.schema.json
│   └── run-report.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── label_master/
    ├── core/
    │   ├── domain/
    │   │   ├── entities.py
    │   │   ├── value_objects.py
    │   │   └── policies.py
    │   └── services/
    │       ├── infer_service.py
    │       ├── validate_service.py
    │       ├── convert_service.py
    │       ├── remap_service.py
    │       └── import_service.py
    ├── adapters/
    │   ├── coco/
    │   │   ├── reader.py
    │   │   ├── writer.py
    │   │   └── detector.py
    │   ├── yolo/
    │   │   ├── reader.py
    │   │   ├── writer.py
    │   │   └── detector.py
    │   └── providers/
    │       ├── kaggle_provider.py
    │       ├── roboflow_provider.py
    │       ├── github_provider.py
    │       └── direct_url_provider.py
    ├── interfaces/
    │   ├── cli/main.py
    │   └── gui/app.py
    ├── infra/
    │   ├── config.py
    │   ├── filesystem.py
    │   ├── locking.py
    │   ├── logging.py
    │   └── reporting.py
    └── reports/schemas.py

tests/
├── unit/
├── integration/
├── contract/
└── fixtures/
    └── us1/
        └── dry_run_samples/
            ├── manifest.template.yaml
            └── <sample_id>/manifest.yaml
```

**Structure Decision**: Single-project Python package was selected to enforce shared core logic and
strict CLI/GUI parity while keeping deployment simple for localhost single-user operation.

## Phase 0 Research Focus

- Confirm Streamlit localhost-only operational pattern and single-user scope handling.
- Choose import protocol policy implementation for `https://`, `http://`, and `file://`.
- Define deterministic write/contention behavior for same-output concurrent runs.
- Define run-artifact compatibility window (v1 current and previous minor versions).
- Validate dry-run strategy using user-provided sample datasets with known format/bbox manifests.
- Validate testing strategy for round-trip bbox invariants and regression safety.

## Phase 1 Design Outputs

- `data-model.md`: canonical entities, validation rules, and state transitions.
- `contracts/`: CLI contract + JSON schemas for run config/report artifacts.
- `quickstart.md`: end-to-end local workflow for infer/convert/import/gui/testing.
- `tests/fixtures/us1/dry_run_samples/manifest.template.yaml`: canonical dry-run sample
  expectation manifest template (known format + known bbox checks).
- Agent context update via `.specify/scripts/bash/update-agent-context.sh codex`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | Plan passes all constitution gates without exceptions |
