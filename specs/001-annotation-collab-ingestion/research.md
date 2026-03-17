# Research: Bounding-Box Conversion and Single-User Web GUI Ingestion

**Feature**: `001-annotation-collab-ingestion`  
**Date**: 2026-03-03

## Decision 1: Runtime and packaging baseline

- **Decision**: Use Python 3.11+ and a single package layout (`src/label_master`).
- **Rationale**: Aligns with constitution and existing design direction; simplifies shared
  library ownership between CLI and GUI.
- **Alternatives considered**:
  - Multi-repo split for CLI/GUI: rejected due to parity drift risk.
  - Python 3.10 baseline: rejected to keep modern typing/runtime features.

## Decision 2: CLI and GUI interface frameworks

- **Decision**: Use Typer for CLI and Streamlit for GUI.
- **Rationale**: Typer provides clear command ergonomics and typed args; Streamlit satisfies
  Python web GUI requirement and rapid localhost workflows.
- **Alternatives considered**:
  - Click-only CLI: rejected (less structured typing ergonomics for this workflow).
  - FastAPI + custom frontend: rejected for v1 complexity overhead.

## Decision 3: GUI access and collaboration scope

- **Decision**: GUI binds localhost only; no built-in authentication/roles in v1; single-user
  scope with future multi-user compatibility via portable run artifacts.
- **Rationale**: Matches clarified scope and minimizes security and operations complexity.
- **Alternatives considered**:
  - Internal-network multi-user without auth: rejected after clarification narrowing to single-user.
  - Public network exposure: rejected due to explicit localhost-only clarification.

## Decision 4: Canonical data model and adapter boundary

- **Decision**: Normalize all annotation formats into a canonical absolute-pixel bbox model in
  core domain before remap/validate/convert operations.
- **Rationale**: Prevents repeated conversion logic and precision drift across adapters.
- **Alternatives considered**:
  - Format-specific processing paths: rejected due to duplicated logic and parity risk.

## Decision 5: Concurrent output-path conflict policy

- **Decision**: Allow concurrent runs targeting the same output path; final artifacts are owned by
  last completed run; both reports record contention and overwrite outcome.
- **Rationale**: Implements clarified behavior while preserving auditability.
- **Alternatives considered**:
  - Reject or queue conflicts: rejected because clarified requirement selected last-write-wins.

## Decision 6: Direct URL protocol policy

- **Decision**: Allow `https://`, `http://`, and `file://` direct URL imports. Emit explicit
  warnings for `http://` and `file://` and record protocol in provenance.
- **Rationale**: Implements clarified policy while preserving safety signaling and traceability.
- **Alternatives considered**:
  - HTTPS-only policy: rejected by clarification.
  - No warnings for non-HTTPS/non-network sources: rejected as too risky.

## Decision 7: Report and configuration contract format

- **Decision**: Use JSON artifacts with explicit schemas for run config and run reports.
- **Rationale**: Enables deterministic machine-readable validation and future backward-compat checks.
- **Alternatives considered**:
  - Ad-hoc JSON without schema: rejected due to weaker compatibility guarantees.
  - YAML-only artifacts: rejected because JSON schema tooling is stronger for validation.

## Decision 8: Validation and fail-closed behavior

- **Decision**: Validate schema, bbox bounds, taxonomy mappings, archive integrity, and import path
  safety before conversion writes. Default policy is fail closed unless explicit override exists.
- **Rationale**: Required by constitution and spec safety expectations.
- **Alternatives considered**:
  - Permissive default with warnings: rejected due to data integrity risk.

## Decision 9: Testing and quality gates

- **Decision**: Use unit + integration + contract + regression tests with `pytest`, `pytest-cov`,
  and Hypothesis for transform invariants; enforce `ruff`, `mypy`, and coverage gates.
- **Rationale**: Matches constitution non-negotiable quality gate requirements.
- **Alternatives considered**:
  - Integration-only strategy: rejected due to weak localization of data-transform regressions.

## Resolved Technical Unknowns

All previously open technical context decisions are resolved in this document. No unresolved
clarification items remain for planning.
