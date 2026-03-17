# Tasks: Streamlit GUI Workflow and Mapping Reliability Enhancements

**Input**: Design documents from `/specs/002-enhance-gui/`  
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md,
data-model.md, contracts/

**Tests**: Tests are REQUIRED for all behavior-changing tasks in this feature.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated
independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel when dependencies are satisfied and files do not overlap.
- **[Story]**: User story label (`US1`, `US2`, `US3`).
- Task descriptions include concrete target files.

## Phase 1: Setup (Shared Preparation)

**Purpose**: Prepare fixtures and contract test scaffolding used by all user stories.

- [X] T001 Create feature task scaffold and checklists directory in `specs/002-enhance-gui/checklists/`
- [X] T002 [P] Add GUI class-map schema fixtures in `tests/fixtures/contracts/{gui-class-map.valid.json,gui-class-map.invalid.json}`
- [X] T003 [P] Add contract test shell for GUI class-map artifact in `tests/contract/test_gui_class_map_schema.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared GUI infrastructure required before user-story implementation.

**⚠️ CRITICAL**: User story tasks start only after this phase is complete.

- [X] T004 Create GUI system action helpers for directory browsing and output opening in `src/label_master/interfaces/gui/system_actions.py`
- [X] T005 [P] Add unit tests for platform/system action fallbacks in `tests/unit/test_gui_system_actions.py`
- [X] T006 Add reusable GUI directory validation helper(s) in `src/label_master/interfaces/gui/app.py`
- [X] T007 [P] Implement contract validation tests for `gui-class-map.schema.json` in `tests/contract/test_gui_class_map_schema.py`

**Checkpoint**: Foundation complete when T004-T007 are done and `pytest tests/unit/test_gui_system_actions.py -q` plus `pytest tests/contract/test_gui_class_map_schema.py -q` pass.

---

## Phase 3: User Story 1 - Select Dataset Directory Without Manual Path Typing (Priority: P1) 🎯 MVP

**Goal**: Provide a working `Browse...` input-directory flow with manual-entry fallback and strict
run gating for invalid directories.

**Independent Test**: Launch GUI, select dataset via browse action (or fallback), and verify
selection/validation state enables preview and run gating correctly.

### Tests for User Story 1 (REQUIRED)

- [X] T008 [P] [US1] Add unit tests for input directory validation outcomes in `tests/unit/test_gui_input_validation.py`
- [X] T009 [P] [US1] Add integration test for browse success/cancel behavior in `tests/integration/test_gui_input_directory_browse.py`
- [X] T010 [P] [US1] Add integration test for browse-unavailable fallback and manual entry continuity in `tests/integration/test_gui_input_directory_browse.py`
- [X] T011 [P] [US1] Add integration test for run blocking on unreadable/non-directory paths in `tests/integration/test_gui_input_directory_blocking.py`

### Implementation for User Story 1

- [X] T012 [US1] Add Dataset-tab `Browse...` action and state keys (`browse_available`, `browse_message`) in `src/label_master/interfaces/gui/app.py`
- [X] T013 [US1] Wire native browse attempt + fallback message path via `src/label_master/interfaces/gui/system_actions.py` and `src/label_master/interfaces/gui/app.py`
- [X] T014 [US1] Enforce existence/type/readability checks in run-blocking validation in `src/label_master/interfaces/gui/app.py`
- [X] T015 [US1] Preserve selected input directory and validation state across tab switches/reruns in `src/label_master/interfaces/gui/app.py`

**Checkpoint**: US1 is complete when browse + fallback work and T008-T011 pass.

---

## Phase 4: User Story 2 - Create Working Label Mappings from the GUI (Priority: P1)

**Goal**: Ensure mapping editor rows are validated, persisted, and applied to conversion behavior.

**Independent Test**: Configure `map` and `drop` rows in GUI, run conversion, and verify output
labels/report totals and persisted map artifact match editor content.

### Tests for User Story 2 (REQUIRED)

- [X] T016 [P] [US2] Extend mapping parse validation tests (duplicates, invalid ints, missing destination) in `tests/unit/test_gui_viewmodels.py`
- [X] T017 [P] [US2] Add integration test that invalid mapping rows disable run action in `tests/integration/test_gui_mapping_blocking.py`
- [X] T018 [P] [US2] Extend GUI mapping persistence/parity integration assertions in `tests/integration/test_gui_inline_mapping_persistence.py`
- [X] T019 [P] [US2] Add contract assertions that persisted `*.gui.class_map.json` matches schema in `tests/contract/test_gui_class_map_schema.py`

### Implementation for User Story 2

- [X] T020 [US2] Normalize mapping rows deterministically across reruns/editor states in `src/label_master/interfaces/gui/app.py`
- [X] T021 [US2] Surface row-level mapping errors and hard-block conversion when errors exist in `src/label_master/interfaces/gui/app.py`
- [X] T022 [US2] Ensure run-specific mapping artifact persistence/rename flow remains schema-compliant in `src/label_master/interfaces/gui/app.py`
- [X] T023 [US2] Keep GUI-to-CLI parity for mapping/unmapped-policy conversion invocation in `src/label_master/interfaces/gui/viewmodels.py` and `src/label_master/interfaces/gui/app.py`
- [X] T024 [US2] Show normalized class-map preview for valid rows in `src/label_master/interfaces/gui/app.py`

**Checkpoint**: US2 is complete when invalid mappings block runs and valid mappings change outputs with passing T016-T019.

---

## Phase 5: User Story 3 - Use a Stitch-Style Guided Run and Report Flow (Priority: P2)

**Goal**: Deliver a guided run/report experience aligned to `docs/stitch_example_gui` interaction
model (not pixel-perfect), including status/progress/summary and output-directory access.

**Independent Test**: Complete one run from Dataset through Review & Run and verify status changes,
summary metrics, and output-access fallback behavior.

### Tests for User Story 3 (REQUIRED)

- [X] T025 [P] [US3] Add integration test for run-state transitions (`idle/running/completed/failed`) in `tests/integration/test_gui_run_summary_status.py`
- [X] T026 [P] [US3] Add integration test for summary metrics rendering from run report totals in `tests/integration/test_gui_run_summary_status.py`
- [X] T027 [P] [US3] Add integration test for output-directory action fallback path visibility in `tests/integration/test_gui_output_access.py`
- [X] T028 [P] [US3] Add unit tests for output open success/failure behavior in `tests/unit/test_gui_system_actions.py`

### Implementation for User Story 3

- [X] T029 [US3] Update tab/section labels and guided flow copy to Stitch-style structure in `src/label_master/interfaces/gui/app.py`
- [X] T030 [US3] Add explicit run-status state and progress presentation in `src/label_master/interfaces/gui/app.py`
- [X] T031 [US3] Render summary cards for processed images, converted labels, warnings/errors in `src/label_master/interfaces/gui/app.py`
- [X] T032 [US3] Implement output-directory access action with fallback messaging/path in `src/label_master/interfaces/gui/system_actions.py` and `src/label_master/interfaces/gui/app.py`
- [X] T033 [US3] Keep run summary artifact references (config path + mapping path) stable in `src/label_master/interfaces/gui/app.py`

**Checkpoint**: US3 is complete when T025-T028 pass and summary/output access behaviors match contract.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final regression protection and documentation alignment across stories.

- [X] T034 [P] Refresh feature guidance in `specs/002-enhance-gui/quickstart.md` to match implemented controls/messages
- [X] T035 [P] Update GUI behavior contract details if implementation-level decisions changed in `specs/002-enhance-gui/contracts/gui-workflow-contract.md`
- [X] T036 Run feature-focused tests and quality gates (`pytest -q`, `ruff check .`, `mypy src`) and record results in `specs/002-enhance-gui/checklists/implementation.md`
- [X] T037 [P] Add SC-001 usability timing evidence (time-to-valid-preview for browse/manual fallback) in `specs/002-enhance-gui/checklists/implementation.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: No dependencies.
- **Phase 2**: Depends on Phase 1 and blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 2; can proceed in parallel with US1 once shared helpers exist.
- **Phase 5 (US3)**: Depends on Phase 2 and reuses US1/US2 run state and artifacts.
- **Phase 6**: Depends on completion of selected user stories.

### User Story Dependencies

- **US1 (P1)**: Independent MVP slice after foundation.
- **US2 (P1)**: Independent behavior slice after foundation; parity verification references existing conversion core behavior.
- **US3 (P2)**: Depends on stable run/mapping artifacts from US1/US2.

### Within Each User Story

- Write tests first and capture failing behavior before implementation.
- Implement helpers/state plumbing before UI wiring.
- Re-run story-specific tests before moving to next story.

### Parallel Opportunities

- Tasks marked `[P]` in each phase can run concurrently when file targets are disjoint.
- US1 and US2 test authoring can run in parallel after foundation.
- Contract test work (`T007`, `T019`) can run in parallel with implementation tasks in other files.

---

## Parallel Example: User Story 1

```bash
Task: "T008 [US1] Unit validation tests in tests/unit/test_gui_input_validation.py"
Task: "T009 [US1] Integration browse tests in tests/integration/test_gui_input_directory_browse.py"
Task: "T011 [US1] Blocking tests in tests/integration/test_gui_input_directory_blocking.py"
```

## Parallel Example: User Story 2

```bash
Task: "T016 [US2] Mapping parse unit tests in tests/unit/test_gui_viewmodels.py"
Task: "T017 [US2] Mapping blocking integration test in tests/integration/test_gui_mapping_blocking.py"
Task: "T019 [US2] Schema contract assertions in tests/contract/test_gui_class_map_schema.py"
```

## Parallel Example: User Story 3

```bash
Task: "T025 [US3] Run-state integration tests in tests/integration/test_gui_run_summary_status.py"
Task: "T027 [US3] Output access integration test in tests/integration/test_gui_output_access.py"
Task: "T028 [US3] Output action unit tests in tests/unit/test_gui_system_actions.py"
```

---

## Implementation Strategy

### MVP First (US1)

1. Complete Phase 1 and Phase 2.
2. Deliver US1 browse/fallback + blocking validation.
3. Validate with US1 tests before proceeding.

### Incremental Delivery

1. Deliver US1 (path selection reliability).
2. Deliver US2 (mapping correctness and artifact parity).
3. Deliver US3 (guided run/report UX and output access).
4. Finish with Phase 6 polish, quality gates, and SC-001 timing evidence.

### Parallel Team Strategy

1. One engineer handles foundational helpers (`T004-T007`).
2. One engineer handles US1 flow/tests.
3. One engineer handles US2 mapping/tests.
4. One engineer handles US3 run/report UX/tests.
