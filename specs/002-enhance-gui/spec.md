# Feature Specification: Streamlit GUI Workflow and Mapping Reliability Enhancements

**Feature Branch**: `002-enhance-gui`  
**Created**: 2026-03-05  
**Status**: Draft  
**Input**: User description: "Feature 002 should enhance the GUI along the lines of the example
at docs/stitch_example_gui, including a button to specify the input directory and fixing the
non-functional label mapping menu."

## Clarifications

### Session 2026-03-05

- Q: Must the GUI match `docs/stitch_example_gui` pixel-for-pixel? -> A: No. It must align to the
  same guided workflow and key interactions while remaining Streamlit-native.
- Q: What is required if a native folder dialog cannot be opened in the runtime environment? -> A:
  Keep manual path entry available as fallback, and show a clear message that browse is unavailable.
- Q: What does "label mapping menu is functional" mean for acceptance? -> A: Mapping edits must
  validate in the UI, persist across tab switches/reruns, be saved as run-specific mapping
  artifacts, and measurably affect conversion output and report totals.
- Q: Should GUI label mapping support non-integer class identifiers in this feature? -> A: No.
  Feature 002 keeps integer class IDs only, matching existing domain contracts.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Dataset Directory Without Manual Path Typing (Priority: P1)

A user configures a conversion run from the GUI by clicking a browse control to choose the input
dataset directory, instead of typing a full filesystem path manually.

**Why this priority**: Directory selection is a primary entry point to every run; if path entry is
friction-heavy, users cannot start confidently.

**Independent Test**: Launch the GUI, click input-directory browse, select a valid fixture
directory, and verify the selected path populates state and enables downstream inference/preview
without manual edits.

**Acceptance Scenarios**:

1. **Given** the Dataset step is visible, **When** the user clicks the input-directory browse
   button and selects an existing directory, **Then** the input path field is populated and the UI
   confirms the directory is valid.
2. **Given** the browse dialog opens, **When** the user cancels selection, **Then** the current
   path remains unchanged and no validation state is incorrectly reset.
3. **Given** the user picks an unreadable or non-directory path, **When** validation runs,
   **Then** the UI shows a blocking, actionable error and run execution remains disabled.

---

### User Story 2 - Create Working Label Mappings from the GUI (Priority: P1)

A user edits source-to-destination label mappings in the GUI and those mappings are actually
applied to conversion output and persisted run artifacts.

**Why this priority**: A non-functional mapping menu can silently produce wrong labels, which is a
high-impact data quality defect.

**Independent Test**: Configure mapping rows in GUI (including `map` and `drop` actions), run a
fixture conversion, and verify output annotations and generated mapping artifact match configured
rows.

**Acceptance Scenarios**:

1. **Given** a mapping row `source_class_id=3, action=map, destination_class_id=10`, **When** a
   conversion run completes, **Then** class `3` annotations are written as class `10` in output.
2. **Given** a mapping row `source_class_id=2, action=drop`, **When** conversion runs, **Then**
   class `2` annotations are excluded and report dropped counts increase accordingly.
3. **Given** invalid mapping rows (duplicate sources, invalid integers, missing destination for
   `map`), **When** the user reaches Review & Run, **Then** the UI shows row-level errors and
   disables run execution.

---

### User Story 3 - Use a Stitch-Style Guided Run and Report Flow (Priority: P2)

A user executes conversion in a guided, tabbed workflow that matches the structure and clarity of
`docs/stitch_example_gui`, including explicit run status, progress, and summary reporting.

**Why this priority**: The visual and flow improvements reduce user confusion and lower setup
mistakes, but are secondary to functional correctness.

**Independent Test**: Complete one end-to-end run using only GUI controls and verify all key state
transitions (setup, mapping, run, summary) are visible and understandable without CLI knowledge.

**Acceptance Scenarios**:

1. **Given** required fields are valid, **When** the user starts conversion, **Then** the run
   section shows real-time or near-real-time status updates (running/completed/failed).
2. **Given** a completed run, **When** the user reviews results, **Then** summary cards show
   processed images, converted labels, and warning/error totals from the run report.
3. **Given** run output exists, **When** the user clicks the output access action, **Then** the UI
   opens the output directory when supported or provides a clear fallback path for manual access.

### Edge Cases

- Input browse control returns a path with spaces or shell-special characters.
- Input directory exists but contains no supported annotation artifacts.
- Input path is valid at selection time but deleted before run starts.
- Mapping table contains blank trailing rows from dynamic row creation.
- Mapping action selector defaults to `map` but destination is omitted.
- User edits mapping rows, switches tabs, and returns; unsaved values must not be lost.
- Report rendering must handle runs with zero converted annotations.
- Output-directory open action fails due to platform restrictions or missing desktop integration.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide an explicit GUI control (button) for selecting the input dataset
  directory without requiring manual full-path typing.
- **FR-002**: System MUST validate selected input directory existence, directory type, and
  readability before enabling conversion execution.
- **FR-003**: System MUST retain directory-selection state across tab navigation and Streamlit
  reruns within the same session.
- **FR-004**: System MUST present a guided, tabbed GUI workflow aligned with
  `docs/stitch_example_gui` structure: setup, format/mapping configuration, and run/report.
- **FR-005**: System MUST provide a label-mapping editor that supports `map` and `drop` actions per
  source class ID.
- **FR-006**: System MUST parse and validate mapping rows with deterministic row-level error
  messages for invalid source IDs, invalid destination IDs, duplicate source IDs, and invalid
  actions.
- **FR-007**: System MUST block conversion execution whenever mapping validation errors exist.
- **FR-008**: System MUST apply valid GUI-defined mappings to conversion output exactly as entered,
  including drop semantics.
- **FR-009**: System MUST persist GUI-generated mappings into a run-specific mapping artifact and
  reference that artifact in exported run configuration/report metadata.
- **FR-010**: System MUST show run status, progress, and a post-run summary that includes processed
  image count, converted annotation count, and warning/error totals.
- **FR-011**: System MUST provide a GUI action to access the output directory and MUST show an
  explicit fallback path when auto-open is unavailable.
- **FR-012**: GUI execution for the same run configuration MUST remain behaviorally equivalent to
  CLI execution in output content and report totals.
- **FR-013**: System MUST define explicit failure behavior for invalid or ambiguous GUI inputs (fail
  closed unless explicitly overridden).
- **FR-014**: System MUST produce deterministic outputs for identical
  inputs/configuration and emit audit-friendly run/report metadata.

### Key Entities *(include if feature involves data)*

- **GuiDirectorySelectionState**: Session-scoped state for selected input/output directories,
  validation state, and last-confirmed valid path.
- **GuiMappingRow**: One editable mapping row containing `source_class_id`, `action` (`map` or
  `drop`), and optional `destination_class_id`.
- **GuiMappingParseResult**: Deterministic parse output containing normalized class map and
  validation errors.
- **GuiRunSummaryView**: Presentation model with run status, progress percentage, core totals, and
  warning/error counts rendered in the report section.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In usability validation with fixture workflows, users can configure input directory
  via browse control and reach a valid preview state in under 60 seconds for at least 90% of
  attempts.
- **SC-002**: For automated GUI integration fixtures covering `map` and `drop` actions, 100% of
  runs produce output annotations consistent with the configured mapping table.
- **SC-003**: In automated negative-path tests, 100% of invalid mapping configurations prevent run
  execution and present at least one actionable row-level validation message.
- **SC-004**: For shared regression fixtures, GUI and CLI runs with equivalent config produce
  matching output files and report summary totals in 100% of test cases.
- **SC-005**: Post-run summary cards and output-directory access action are available for 100% of
  successful runs in GUI regression tests.
