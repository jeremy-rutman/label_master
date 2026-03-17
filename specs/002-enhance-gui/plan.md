# Implementation Plan: Streamlit GUI Workflow and Mapping Reliability Enhancements

**Branch**: `002-enhance-gui` | **Date**: 2026-03-05 | **Spec**: [/home/jeremy/Documents/label_master/specs/002-enhance-gui/spec.md](/home/jeremy/Documents/label_master/specs/002-enhance-gui/spec.md)
**Input**: Feature specification from `/specs/002-enhance-gui/spec.md`

## Summary

Enhance the existing Streamlit GUI to provide a guided Stitch-style workflow with two critical
functional fixes: a browse-button path selection flow for input directory selection (with robust
fallback when native dialogs are unavailable) and a fully reliable label mapping editor whose
validated rows persist and directly affect conversion outputs/reports. Preserve CLI/GUI parity by
reusing current core services and run artifact formats.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: Streamlit (GUI), Typer (CLI parity surface), Pydantic v2
(run/report/view-model validation), PyYAML (mapping/config IO), Pillow (preview rendering), pytest
+ pytest-cov (tests), ruff + mypy (quality gates)  
**Storage**: Local filesystem only (dataset paths, generated mapping artifacts under `reports/`,
run config JSON, run report JSON)  
**Testing**: pytest unit + integration + targeted contract/schema checks for GUI artifacts; parity
regression checks between GUI and CLI outputs  
**Target Platform**: Localhost-only execution on Linux/macOS workstations; degraded path-picker mode
must work in headless/no-display environments  
**Project Type**: Single Python package with shared core library and Streamlit GUI adapter  
**Performance Goals**: Input-path validation and mapping parse feedback <250ms on typical local
datasets/mapping sizes; no measurable conversion regression (>5%) relative to baseline when GUI
enhancements are enabled  
**Constraints**: Preserve core conversion logic boundaries; GUI remains localhost-only; integer
class IDs for mapping; fail-closed run gating for invalid paths/mappings; deterministic run outputs
and audit metadata unchanged  
**Scale/Scope**: Single-user GUI sessions, up to existing v1 dataset sizes (~50k images / ~100k
annotations) with no scope expansion into multi-user auth/roles

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Gate

- **Core Library and Canonical Model First**: PASS
  - No conversion/remap business rules move into GUI; GUI continues to call shared core services.
- **CLI-Complete and GUI-Parity Interfaces**: PASS
  - Mapping/path selections still compile into the same conversion inputs and run artifacts as CLI.
- **Safety, Determinism, and Auditability**: PASS
  - Invalid directory/mapping states block execution; run/report artifacts and deterministic write
    behavior remain unchanged.
- **Test-Driven Quality Gates**: PASS
  - Plan includes unit/integration regression tests for browse fallback, mapping correctness, and
    GUI/CLI parity.
- **Spec-Driven Delivery and Change Control**: PASS
  - Work maps to US1-US3, FR-001..FR-014, and SC-001..SC-005 in spec 002.

### Post-Design Gate (Phase 1 Re-check)

- **Core Library and Canonical Model First**: PASS
  - `data-model.md` keeps GUI state entities separate from core annotation domain entities.
- **CLI-Complete and GUI-Parity Interfaces**: PASS
  - `contracts/gui-workflow-contract.md` explicitly binds GUI run actions to existing CLI-equivalent
    conversion parameters.
- **Safety, Determinism, and Auditability**: PASS
  - Contracts include blocking conditions for invalid mappings/paths and preserve mapping artifact
    traceability.
- **Test-Driven Quality Gates**: PASS
  - `quickstart.md` defines regression commands covering mapping/output parity.
- **Spec-Driven Delivery and Change Control**: PASS
  - Phase outputs are scoped to clarified requirements only; no hidden behavior expansion.

## Project Structure

### Documentation (this feature)

```text
specs/002-enhance-gui/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── gui-workflow-contract.md
│   └── gui-class-map.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── label_master/
    ├── interfaces/
    │   └── gui/
    │       ├── app.py
    │       └── viewmodels.py
    ├── infra/
    │   ├── filesystem.py
    │   └── reporting.py
    └── interfaces/
        └── cli/main.py

tests/
├── unit/
│   └── test_gui_viewmodels.py
├── integration/
│   ├── test_gui_inline_mapping_persistence.py
│   └── test_gui_contention_last_write_wins.py
└── contract/
```

**Structure Decision**: Keep the existing single-package architecture and implement GUI-only
enhancements in `interfaces/gui` while preserving shared core service invocation paths and existing
artifact schemas.

## Phase 0 Research Focus

- Validate feasible input-directory browse mechanisms in Streamlit and define fallback behavior for
  no-display/headless runtime.
- Define deterministic state model for mapping editor rows across reruns and tab navigation.
- Define run-status/progress and summary presentation strategy using existing report payloads.
- Define cross-platform output-directory access behavior and non-fatal fallback messaging.
- Define regression strategy for proving mapping-menu functionality and GUI/CLI parity.

## Phase 1 Design Outputs

- `research.md`: implementation decisions and rejected alternatives for browse, mapping reliability,
  and run/report UX.
- `data-model.md`: GUI state/value objects and validation invariants for directory selection,
  mapping rows, run summary, and output access.
- `contracts/gui-workflow-contract.md`: normative GUI behavior and run-gating contract.
- `contracts/gui-class-map.schema.json`: schema for generated GUI mapping artifact payload.
- `quickstart.md`: developer/operator flow for running GUI enhancements and executing quality gates.
- Agent context update via `.specify/scripts/bash/update-agent-context.sh codex`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | Plan satisfies constitution gates without exceptions |
