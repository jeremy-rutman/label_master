# Tasks: Bounding-Box Conversion and Single-User Web GUI Ingestion

**Input**: Design documents from `/specs/001-annotation-collab-ingestion/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED for behavior-changing stories and bug fixes.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel only when task dependencies are already satisfied and write-target files do not overlap
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions
- For brace-expanded file groups (for example `{reader.py,writer.py,detector.py}`), task completion requires required behavior in each listed file
- Test tasks MUST capture fail-before and pass-after evidence in review artifacts for behavior changes

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and baseline tooling.

- [X] T001 Create package skeleton in `src/label_master/{core/domain,core/services,adapters/coco,adapters/yolo,adapters/providers,interfaces/cli,interfaces/gui,infra,reports}/__init__.py`
- [X] T002 Create Python project configuration and CLI entrypoint in `pyproject.toml`
- [X] T003 [P] Configure linting, typing, and pytest defaults in `pyproject.toml`
- [X] T004 [P] Create baseline test package files in `tests/{unit,integration,contract,fixtures}/__init__.py`
- [X] T005 [P] Add development command shortcuts in `Makefile`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story work.

**⚠️ CRITICAL**: No user story implementation starts until this phase is complete.

- [X] T006 Implement canonical entities and shared enums in `src/label_master/core/domain/entities.py`
- [X] T007 [P] Implement value objects and typed error primitives in `src/label_master/core/domain/value_objects.py`
- [X] T008 [P] Implement policy models (unmapped, validation, inference thresholds) in `src/label_master/core/domain/policies.py`
- [X] T009 Implement run/report schema models in `src/label_master/reports/schemas.py`
- [X] T010 Implement safe filesystem/path helpers in `src/label_master/infra/filesystem.py`
- [X] T011 [P] Implement structured logging utilities in `src/label_master/infra/logging.py`
- [X] T012 [P] Implement config loading and override precedence in `src/label_master/infra/config.py`
- [X] T013 Implement output-path contention lock manager with last-write-wins metadata in `src/label_master/infra/locking.py`
- [X] T014 Implement report writer and artifact persistence utilities in `src/label_master/infra/reporting.py`
- [X] T015 [P] Add foundational unit tests for entities and policies in `tests/unit/test_domain_foundation.py`
- [X] T016 [P] Add foundational unit tests for path safety, locking, and reporting in `tests/unit/test_infra_foundation.py`
- [X] T017 Add JSON-schema validation fixtures in `tests/fixtures/contracts/`

**Checkpoint**: Foundation is complete only when ALL conditions below are true.

- T006-T017 are complete.
- Foundational unit tests in T015-T016 pass.
- Contract fixtures in T017 validate against current schemas.
- Baseline quality commands (`ruff check .`, `mypy src`, and `pytest -q`) pass on the branch.

---

## Phase 3: User Story 1 - Convert and Remap via CLI (Priority: P1) 🎯 MVP

**Goal**: Deliver deterministic COCO/YOLO infer-validate-convert-remap workflows via CLI.

**Independent Test**: Execute `annobox infer|validate|convert|remap` on fixtures and verify deterministic outputs, drop/unmapped counts, and required exit codes.

### Tests for User Story 1 (REQUIRED)

- [X] T018 [P] [US1] Add CLI contract tests for `infer|validate|convert|remap` in `tests/contract/test_cli_conversion_contract.py`
- [X] T019 [P] [US1] Add integration test for COCO->YOLO conversion with remap/drop in `tests/integration/test_cli_convert_remap.py`
- [X] T020 [P] [US1] Add integration test for YOLO->COCO deterministic ordering in `tests/integration/test_cli_roundtrip_ordering.py`
- [X] T021 [P] [US1] Add property test for bbox transform invariants in `tests/unit/test_bbox_invariants.py`
- [X] T062 [P] [US1] Add dry-run integration tests using user-provided known-bbox sample datasets and YAML manifests to verify inferred format, bbox-check outcomes, and zero output writes in `tests/integration/test_cli_dry_run_known_bbox_samples.py`

### Implementation for User Story 1

- [X] T022 [P] [US1] Implement COCO adapter reader/writer/detector in `src/label_master/adapters/coco/{reader.py,writer.py,detector.py}`
- [X] T023 [P] [US1] Implement YOLO adapter reader/writer/detector in `src/label_master/adapters/yolo/{reader.py,writer.py,detector.py}`
- [X] T024 [US1] Implement format inference orchestration in `src/label_master/core/services/infer_service.py`
- [X] T025 [US1] Implement schema and bbox validation orchestration in `src/label_master/core/services/validate_service.py`
- [X] T026 [US1] Implement remap logic with `error|drop|identity` policy in `src/label_master/core/services/remap_service.py`
- [X] T027 [US1] Implement convert orchestration and deterministic writes in `src/label_master/core/services/convert_service.py`
- [X] T028 [US1] Implement CLI commands and exit-code mapping in `src/label_master/interfaces/cli/main.py`
- [X] T029 [US1] Wire run-config/report artifact generation in `src/label_master/interfaces/cli/main.py`
- [X] T030 [US1] Add US1 fixture datasets and mapping files in `tests/fixtures/us1/`
- [X] T063 [US1] Implement dry-run execution path that performs infer/validate/remap simulation with zero converted-annotation writes in `src/label_master/core/services/convert_service.py`
- [X] T064 [US1] Implement expected-format and bbox-manifest verification for dry-run sample datasets, including YAML manifest parsing/validation, in `src/label_master/core/services/validate_service.py`
- [X] T065 [US1] Add user-provided dry-run sample datasets and expected manifests (seeded from `tests/fixtures/us1/dry_run_samples/manifest.template.yaml`) in `tests/fixtures/us1/dry_run_samples/`

**Checkpoint**: US1 is independently functional and testable.

---

## Phase 4: User Story 2 - Infer Format and Run Conversion from Localhost Web GUI (Priority: P1)

**Goal**: Deliver Streamlit localhost GUI parity for infer/convert flows, including contention reporting.

**Independent Test**: Run GUI on localhost, perform conversion workflow, export run config, and reproduce equivalent output via CLI.

### Tests for User Story 2 (REQUIRED)

- [X] T031 [P] [US2] Add integration test for localhost-only GUI binding in `tests/integration/test_gui_localhost_only.py`
- [X] T032 [P] [US2] Add integration test for GUI/CLI output parity in `tests/integration/test_gui_cli_parity.py`
- [X] T033 [P] [US2] Add integration test for same-output-path contention behavior in `tests/integration/test_gui_contention_last_write_wins.py`

### Implementation for User Story 2

- [X] T034 [US2] Implement GUI viewmodels that call shared services in `src/label_master/interfaces/gui/viewmodels.py`
- [X] T035 [US2] Implement Streamlit workflow screens in `src/label_master/interfaces/gui/app.py`
- [X] T036 [US2] Implement GUI run-config export compatible with run-config schema in `src/label_master/interfaces/gui/app.py`
- [X] T037 [US2] Wire contention events from lock manager to run reports in `src/label_master/core/services/convert_service.py`
- [X] T038 [US2] Add GUI parity and contention fixtures in `tests/fixtures/us2/`

**Checkpoint**: US1 and US2 both work independently; GUI remains localhost-only.

---

## Phase 5: User Story 3 - Import Datasets from External Sources (Priority: P2)

**Goal**: Support Kaggle/Roboflow/GitHub/direct URL imports with provenance, protocol warnings, and fail-closed validation.

**Independent Test**: Import fixtures for each source type, verify warnings for `http://`/`file://`, and confirm provenance + integrity behavior.

### Tests for User Story 3 (REQUIRED)

- [X] T039 [P] [US3] Add CLI import contract tests for provider arguments in `tests/contract/test_cli_import_contract.py`
- [X] T040 [P] [US3] Add integration tests for provider import flows in `tests/integration/test_import_providers.py`
- [X] T041 [P] [US3] Add integration tests for `http://` and `file://` warning behavior in `tests/integration/test_import_protocol_warnings.py`
- [X] T042 [P] [US3] Add unit tests for `file://` path traversal protections in `tests/unit/test_import_path_safety.py`

### Implementation for User Story 3

- [X] T043 [P] [US3] Implement Kaggle/Roboflow/GitHub provider adapters in `src/label_master/adapters/providers/{kaggle_provider.py,roboflow_provider.py,github_provider.py}`
- [X] T044 [US3] Implement direct URL provider with `https|http|file` handling and warnings in `src/label_master/adapters/providers/direct_url_provider.py`
- [X] T045 [US3] Implement import orchestration and integrity/schema gating in `src/label_master/core/services/import_service.py`
- [X] T046 [US3] Integrate import command and provenance reporting in `src/label_master/interfaces/cli/main.py`
- [X] T047 [US3] Add provider/import fixtures and malformed archive cases in `tests/fixtures/us3/`

**Checkpoint**: US3 import workflows run independently and integrate with conversion pipeline.

---

## Phase 6: User Story 4 - Review Quality and Change Traceability (Priority: P3)

**Goal**: Enforce CI quality gates and traceability requirements for behavior-changing changes.

**Independent Test**: Simulate PR validation and confirm failure when required spec/test evidence is missing.

### Tests for User Story 4 (REQUIRED)

- [X] T048 [P] [US4] Add traceability policy test for linked spec evidence in `tests/contract/test_spec_traceability_policy.py`
- [X] T049 [P] [US4] Add regression-policy test for behavior-changing fixes in `tests/integration/test_regression_policy_gate.py`

### Implementation for User Story 4

- [X] T050 [US4] Implement spec-traceability checker script in `scripts/ci/check_spec_traceability.py`
- [X] T051 [US4] Implement regression-test evidence checker in `scripts/ci/check_regression_tests.py`
- [X] T052 [US4] Add CI workflow for lint/type/test/coverage/traceability gates in `.github/workflows/ci.yml`
- [X] T053 [US4] Add review evidence template in `.github/pull_request_template.md`
- [X] T054 [US4] Document review and traceability policy in `docs/review-quality-gates.md`

**Checkpoint**: Review and governance checks are automated and testable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening across all stories.

- [X] T055 [P] Update design documentation with final behavior notes in `docs/bbox-annotation-converter-design-spec.md`
- [X] T056 Add performance smoke test for conversion and inference thresholds in `tests/integration/test_performance_smoke.py`
- [X] T057 [P] Validate and refresh execution steps in `specs/001-annotation-collab-ingestion/quickstart.md`
- [X] T058 Run full quality gate commands and record outcomes in `specs/001-annotation-collab-ingestion/checklists/implementation.md`
- [X] T059 [P] Add contract compatibility tests for run artifacts across current and previous v1 minor schema versions (N and N-1) in `tests/contract/test_run_artifact_compatibility.py`
- [X] T060 Add inference accuracy benchmark gate for SC-002 (>=95% top-1) in `tests/integration/test_inference_accuracy_benchmark.py`
- [X] T061 Add conversion success-rate benchmark gate for SC-003 (>=95%) in `tests/integration/test_conversion_success_rate.py`
- [X] T066 Implement run-artifact schema version negotiation/upgrade shims for current and previous v1 minor versions in `src/label_master/reports/schemas.py` and `src/label_master/infra/config.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 2 and reuses US1 conversion/report interfaces for parity checks.
- **Phase 5 (US3)**: Depends on Phase 2 and reuses US1 validation/report components.
- **Phase 6 (US4)**: Depends on Phases 3-5 outputs to enforce policy gates on real workflows.
- **Phase 7 (Polish)**: Depends on completion of selected user stories.

### User Story Dependencies

- **US1 (P1)**: Independent after foundational work; recommended MVP slice.
- **US2 (P1)**: Depends on shared services and CLI contract behavior from US1 for parity verification.
- **US3 (P2)**: Depends on foundational infra and shared validation/reporting services; can run in parallel with US2 after US1 service contracts stabilize.
- **US4 (P3)**: Depends on implemented test suites and CI commands from US1-US3.

### Within Each User Story

- Tests first (must fail before implementation): capture failing command evidence before behavior code changes and passing evidence after.
- Adapters/models before service orchestration.
- Service orchestration before interface wiring.
- Fixtures and docs updated before story checkpoint.

### Parallel Opportunities

- Setup tasks marked `[P]` can run simultaneously.
- Foundational tasks marked `[P]` can run simultaneously.
- In each user story, `[P]` tests can run in parallel.
- Adapter/provider implementations marked `[P]` can run in parallel when touching different files.

---

## Parallel Example: User Story 1

```bash
Task: "T018 [US1] CLI contract tests in tests/contract/test_cli_conversion_contract.py"
Task: "T019 [US1] Integration test in tests/integration/test_cli_convert_remap.py"
Task: "T020 [US1] Integration test in tests/integration/test_cli_roundtrip_ordering.py"
Task: "T021 [US1] Property test in tests/unit/test_bbox_invariants.py"

Task: "T022 [US1] COCO adapter in src/label_master/adapters/coco/{reader.py,writer.py,detector.py}"
Task: "T023 [US1] YOLO adapter in src/label_master/adapters/yolo/{reader.py,writer.py,detector.py}"
```

## Parallel Example: User Story 2

```bash
Task: "T031 [US2] Localhost binding test in tests/integration/test_gui_localhost_only.py"
Task: "T032 [US2] GUI/CLI parity test in tests/integration/test_gui_cli_parity.py"
Task: "T033 [US2] Contention test in tests/integration/test_gui_contention_last_write_wins.py"
```

## Parallel Example: User Story 3

```bash
Task: "T039 [US3] CLI import contract tests in tests/contract/test_cli_import_contract.py"
Task: "T040 [US3] Provider flow tests in tests/integration/test_import_providers.py"
Task: "T041 [US3] Protocol warning tests in tests/integration/test_import_protocol_warnings.py"
Task: "T042 [US3] Path-safety tests in tests/unit/test_import_path_safety.py"

Task: "T043 [US3] Provider adapters in src/label_master/adapters/providers/{kaggle_provider.py,roboflow_provider.py,github_provider.py}"
```

## Parallel Example: User Story 4

```bash
Task: "T048 [US4] Traceability policy test in tests/contract/test_spec_traceability_policy.py"
Task: "T049 [US4] Regression gate test in tests/integration/test_regression_policy_gate.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup).
2. Complete Phase 2 (Foundational).
3. Complete Phase 3 (US1).
4. Validate US1 contract/integration/property tests.
5. Demo CLI-first conversion/remap workflow.

### Incremental Delivery

1. Deliver US1 (CLI conversion/remap core).
2. Deliver US2 (localhost Streamlit parity).
3. Deliver US3 (external imports + provenance warnings).
4. Deliver US4 (quality/traceability automation).
5. Finish polish and performance checks.

### Parallel Team Strategy

1. Team completes Setup + Foundational together.
2. Developer A: US1 service/core tasks.
3. Developer B: US2 GUI tasks after US1 contracts stabilize.
4. Developer C: US3 import-provider tasks after foundational infra is ready.
5. Developer D: US4 CI/policy automation after tests exist.

---

## Requirement Quality Addendum (Checklist Implementation)

### Objective Definitions and Acceptance Boundaries

- **Equivalent output (SC-004)**: Same normalized artifact set for the same run configuration, where
  - converted annotation files are byte-equivalent after deterministic ordering rules are applied,
  - run report totals and categorical counters are identical, and
  - schema version and run identifiers are valid per run-report contract.
- **Deterministic ordering (FR-018)**: Stable ordering by canonical path and record identifier for identical inputs/configuration.
- **Success-rate benchmark (SC-003)**: `successful_runs / validated_runs >= 0.95` over the benchmark fixture suite; denominator excludes intentionally invalid fixtures.
- **Foundation complete (Phase 2)**: all objective criteria listed in the Phase 2 checkpoint above.
- **Phase 7 complete**: all tasks T055-T061 and T066 complete and quality gates plus policy scripts pass.

### Quality Gate Command Set (T058)

The canonical command set for T058 and CI parity is:

```bash
ruff check .
mypy src
pytest -q
pytest --cov=src/label_master --cov-report=term-missing
python scripts/ci/check_spec_traceability.py --evidence <path>
python scripts/ci/check_regression_tests.py --evidence <path>
```

Command outcomes MUST be recorded in `specs/001-annotation-collab-ingestion/checklists/implementation.md`.
Changes to this command set require updates to both T058 notes and `.github/workflows/ci.yml`.

### FR-to-Task Traceability Matrix

| Requirement | Task Coverage |
|-------------|---------------|
| FR-001 | T022, T023, T027, T019, T020 |
| FR-002 | T024, T025, T026, T028, T029, T018 |
| FR-003 | T034, T035, T032 |
| FR-003a | T035 |
| FR-003b | T031, T035 |
| FR-004 | T024, T018 |
| FR-005 | T024, T025, T018 |
| FR-006 | T026, T019 |
| FR-007 | T026, T019 |
| FR-008 | T026, T018, T019 |
| FR-009 | T029, T037 |
| FR-010 | T063, T062 |
| FR-010a | T062, T064, T065 |
| FR-010b | T064, T065, T062 |
| FR-011 | T025, T018 |
| FR-012 | T043, T040 |
| FR-013 | T043, T040 |
| FR-014 | T043, T040 |
| FR-015 | T044, T040 |
| FR-015a | T041, T044, T046 |
| FR-016 | T045, T046, T040 |
| FR-017 | T025, T026, T029 |
| FR-018 | T027, T020 |
| FR-019 | T036, T032 |
| FR-019a | T033, T037 |
| FR-019b | T009, T059, T066 |
| FR-020 | T028, T018 |
| FR-021 | T014, T029 |

### SC-to-Task Traceability Matrix

| Success Criterion | Task Coverage |
|-------------------|---------------|
| SC-001 | T019, T026, T027 |
| SC-002 | T024, T060 |
| SC-003 | T025, T027, T061 |
| SC-004 | T032, T034, T036 |
| SC-005 | T040, T043, T045, T046 |
| SC-006 | T015, T016, T018-T042, T058 |
| SC-007 | T048, T050, T053 |
| SC-008 | T033, T037 |
| SC-009 | T031, T035 |
| SC-010 | T041, T044, T046 |
| SC-011 | T062, T063, T064, T065 |

### Acceptance Scenario Mapping

| Scenario Group | Scenario Summary | Task Coverage |
|----------------|------------------|---------------|
| US1-1 | COCO->YOLO conversion with remap and dropped-label reporting | T019, T026, T027, T029 |
| US1-2 | Unmapped policy `error` exits non-zero and reports unmapped IDs | T018, T026, T028 |
| US1-3 | Unmapped policy `drop` excludes labels with drop counts | T019, T026, T029 |
| US1-4 | Dry-run known-sample verification with zero converted output writes | T062, T063, T064, T065 |
| US2-1 | GUI infer shows candidate/confidence/evidence/ambiguity | T034, T035, T032 |
| US2-2 | Ambiguous inference blocks conversion unless resolved | T034, T035, T032 |
| US2-3 | GUI export replayable by CLI with equivalent output | T032, T036 |
| US2-4 | Same-output contention last-write-wins and both reports note overwrite | T033, T037 |
| US2-5 | GUI rejects non-localhost access | T031, T035 |
| US3-1 | Valid source import records provenance | T040, T043, T045, T046 |
| US3-2 | Inaccessible/invalid source fails closed with actionable errors | T040, T045, T046 |
| US3-3 | Imported datasets run through same conversion pipeline | T040, T045, T027 |
| US3-4 | `http://`/`file://` warnings shown and protocol recorded | T041, T044, T046 |
| US4-1 | Behavior-changing PR must pass tests/static checks | T049, T052 |
| US4-2 | Bug-fix review requires regression evidence | T049, T051, T053 |

### Edge Case Mapping

| Edge Case | Task Coverage | Status |
|-----------|---------------|--------|
| Mixed-format directories (COCO + YOLO) | T018, T024, T025 | In scope |
| Missing/unreadable image files for bbox bounds checks | T025, T062, T064 | In scope |
| YOLO rows invalid token count/non-numeric/out-of-range | T018, T025 | In scope |
| COCO annotations missing category IDs/malformed bbox arrays | T018, T025 | In scope |
| Class maps with destination IDs outside taxonomy | T019, T026 | In scope |
| Duplicate/conflicting imported files across sources | T040, T045, T047 | In scope |
| Unsupported compression/directory nesting depth in archives | T040, T045, T047 | In scope |
| Interrupted imports leave partial datasets | T040, T045 | In scope |
| Concurrent writes to same output path | T033, T037 | In scope |
| GUI bind misconfigured to non-localhost | T031, T035 | In scope |
| Future multi-user editing conflicts | Deferred to future feature | Out of scope for v1 |
| `file://` imports to disallowed paths/path traversal | T041, T042, T044 | In scope |
| Dry-run accidentally writes converted output | T062, T063 | In scope |

### Multi-File Task Boundary Clarifications

| Task | Required Minimum Outputs |
|------|---------------------------|
| T022 | `reader.py`, `writer.py`, and `detector.py` all implemented for COCO with tests exercising each capability |
| T023 | `reader.py`, `writer.py`, and `detector.py` all implemented for YOLO with tests exercising each capability |
| T043 | `kaggle_provider.py`, `roboflow_provider.py`, and `github_provider.py` each include retrieval + provenance hooks used by import orchestration |

### Governance Gate Pass/Fail Criteria

- **T050 (`check_spec_traceability.py`)** fails when review evidence omits either spec references or test references for behavior changes.
- **T051 (`check_regression_tests.py`)** fails when bug-fix evidence does not include regression-test linkage to corrected failure mode.
- **T052 (CI workflow)** must execute lint, type check, tests, coverage gate, traceability check, and regression-policy check.
- **T053 (PR template)** must require links to spec artifacts and test evidence paths for behavior-changing changes.

### External Dependencies and Execution Prerequisites

- Kaggle import coverage requires valid environment credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or mocked fixtures.
- Roboflow and GitHub import coverage require network access or fixture-backed simulation in CI.
- Direct URL coverage requires controlled `https://`, `http://`, and `file://` fixtures plus path-safety checks.
- Provider integration tests require reproducible fixture archives and known checksums where integrity is asserted.
- Localhost GUI tests require loopback networking support in the execution environment.

### Documentation and Contract Synchronization Rules

| Artifact | Synchronized With Tasks |
|----------|--------------------------|
| `contracts/cli-contract.md` | T028, T046, T057 |
| `contracts/run-config.schema.json` and `contracts/run-report.schema.json` | T009, T036, T059, T066 |
| `docs/review-quality-gates.md` and PR template | T050-T054 |
| `quickstart.md` | T057 plus command-set changes from T058 |

If any of the synchronized tasks change behavior, the paired artifact above MUST be updated in the same PR.

### Overlap Resolution and Ownership Boundaries

| Potential Overlap | Boundary Rule |
|-------------------|---------------|
| T024 vs T025 | T024 owns inference ranking/ambiguity; T025 owns schema+bbox validation and gating decisions |
| T029 vs T037 | T029 owns generic CLI run artifact wiring; T037 owns contention event integration into reports |
| T044 vs T045 | T044 owns direct URL transport/protocol warning behavior; T045 owns import orchestration/fail-closed policy |
| T050 vs T051 | T050 enforces spec-to-test traceability; T051 enforces bug-fix regression evidence policy |

### Deferred Scope and Explicit Non-Goals

- Multi-user collaboration, authentication, and role-based authorization remain out of scope for v1.
- Non-bbox annotations (segmentation, keypoints) remain out of scope for this feature branch.
- Any future expansion beyond N/N-1 run-artifact compatibility is deferred to a future spec amendment.

### tasks.md Synchronization Cadence

- On any approved change to `spec.md` or `plan.md`, update affected task IDs and traceability tables in `tasks.md` within one working day.
- Any new FR/SC added to `spec.md` requires a same-PR update to the FR/SC traceability matrices above.
- CI/policy command changes require same-PR updates to T058 command set documentation and implementation checklist evidence format.

---

## Notes

- All tasks follow required checklist format with IDs, optional `[P]`, and `[USx]` labels for story phases.
- Story tasks include explicit file paths to support direct execution.
- Suggested MVP scope is Phase 3 (US1) after Setup and Foundational phases.
