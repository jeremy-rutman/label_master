# Research: Streamlit GUI Workflow and Mapping Reliability Enhancements

**Feature**: `002-enhance-gui`  
**Date**: 2026-03-05

## Decision 1: Input-directory browse interaction strategy

- **Decision**: Add an explicit `Browse...` action in the Dataset step that attempts native
  directory selection where runtime supports it, while preserving manual text input as a guaranteed
  fallback.
- **Rationale**: Satisfies FR-001 usability requirement without making GUI functionality depend on
  display-server availability.
- **Alternatives considered**:
  - Manual path input only: rejected because it does not satisfy feature requirement.
  - Custom Streamlit frontend component for directory dialogs: rejected for complexity and
    maintenance overhead in v1 scope.

## Decision 2: Fallback behavior when native browse is unavailable

- **Decision**: If browse cannot open (headless runtime, missing toolkit, permission issue), show a
  non-blocking message and keep the path input editable for manual entry.
- **Rationale**: Implements clarification from 2026-03-05 and avoids regressions for remote/headless
  development setups.
- **Alternatives considered**:
  - Hard fail when browse is unavailable: rejected as fragile and user-hostile.
  - Silent failure: rejected due to diagnosability issues.

## Decision 3: Mapping editor state authority

- **Decision**: Session-state normalized rows remain the canonical GUI mapping source. Parsing to
  `class_map` happens deterministically on every relevant rerun (mapping tab + review tab).
- **Rationale**: Prevents stale editor snapshots and ensures FR-006/FR-007 run-gating consistency.
- **Alternatives considered**:
  - Parse only on run button click: rejected because users receive late feedback and higher error
    risk.
  - Store only parsed map and drop raw rows: rejected because row-level UX feedback becomes weak.

## Decision 4: Definition of "mapping menu functional"

- **Decision**: A mapping is considered functional only when it is validated, persisted to a
  run-specific artifact, and measurably changes conversion output/report counts.
- **Rationale**: Matches clarification and closes the gap between UI appearance and conversion
  behavior.
- **Alternatives considered**:
  - Treat visual editor updates as success without output validation: rejected due to silent data
    quality risk.

## Decision 5: Stitch-style workflow implementation scope

- **Decision**: Adopt Stitch-style guided flow semantics (setup, mapping, run/report) rather than
  pixel-perfect UI replication.
- **Rationale**: Clarified requirement explicitly targets interaction model, not exact styling.
- **Alternatives considered**:
  - Pixel-perfect recreation of HTML mock: rejected as unnecessary and brittle within Streamlit.

## Decision 6: Run status, progress, and summary presentation

- **Decision**: Use existing run report payload to drive summary metrics and show explicit run state
  transitions (`idle`, `running`, `completed`, `failed`) in Review & Run.
- **Rationale**: Reuses canonical run/report artifacts and avoids duplicate bookkeeping logic.
- **Alternatives considered**:
  - Separate GUI-only result model disconnected from report: rejected due to parity drift risk.

## Decision 7: Output-directory access behavior

- **Decision**: Provide best-effort platform-specific open-directory action, and always show
  deterministic fallback path if auto-open is unavailable or fails.
- **Rationale**: Meets FR-011 while remaining safe across Linux/macOS and sandboxed environments.
- **Alternatives considered**:
  - Require auto-open success: rejected due to environment variability.
  - Omit open action entirely: rejected because it misses explicit user-facing workflow requirement.

## Decision 8: Test strategy for this feature

- **Decision**: Add targeted unit tests for browse fallback and mapping-row validation/run gating,
  plus integration coverage proving GUI mapping artifacts produce expected output parity with CLI.
- **Rationale**: Aligns with constitution quality gates and SC-002/SC-004.
- **Alternatives considered**:
  - Manual QA only: rejected because it cannot enforce regression safety in CI.

## Resolved Technical Unknowns

All planning-level technical unknowns for Feature 002 are resolved in this document. No open
clarification markers remain.
