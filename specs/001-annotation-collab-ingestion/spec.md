# Feature Specification: Bounding-Box Conversion and Single-User Web Ingestion

**Feature Branch**: `001-annotation-collab-ingestion`  
**Created**: 2026-03-03  
**Status**: Draft  
**Input**: User description: "Bounding-box annotation conversion tool with full CLI control, GUI usability, format inference, class remapping with drop semantics, single-user localhost web GUI for v1, and data pulls from Kaggle, Roboflow, GitHub, and user-specified URLs."
**Branch Naming Note**: The branch slug retains initial "collab" wording for traceability; v1 scope in this spec is single-user web ingestion.

## Clarifications

### Session 2026-03-03

- Q: What is the GUI access model? -> A: No user roles or authentication.
- Q: What is the concurrent-run conflict policy for the same output path? -> A: Allow parallel
  runs; last write wins.
- Q: What is the GUI network exposure boundary? -> A: Bind localhost only.
- Q: What is v1 collaboration scope? -> A: Single-user web GUI now; multi-user support later.
- Q: Which direct URL protocols are allowed for import? -> A: Allow `https://`, `http://`, and
  `file://`; show warnings for `http://` and `file://`.

### Session 2026-03-04

- Q: How should dry-run correctness be checked? -> A: Use user-provided sample datasets with known
  format and bbox locations; verify inferred format and bbox checks in dry-run without writing
  converted annotation outputs.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Convert and Remap via CLI (Priority: P1)

A dataset engineer runs command-line workflows to convert annotation datasets between supported
formats and remap class IDs, including intentionally dropping selected labels.

**Why this priority**: CLI-first automation is required for reproducible ML workflows and is the
core value of the product.

**Independent Test**: Can be fully tested by running conversion and remap commands on fixture
datasets, and running dry-run on user-provided known-bbox samples to verify inferred format and
bbox checks while producing no converted output files.

**Acceptance Scenarios**:

1. **Given** a valid COCO dataset and a valid class map, **When** the user runs conversion to
   YOLO with remapping, **Then** output files are produced with mapped classes and a report shows
   before/after class counts and dropped labels.
2. **Given** an unmapped source class and unmapped policy `error`, **When** conversion runs,
   **Then** the run exits with a non-zero status and reports unmapped class IDs.
3. **Given** unmapped policy `drop`, **When** conversion runs, **Then** annotations from unmapped
   classes are excluded and the run report lists how many were dropped.
4. **Given** a dry-run uses a sample dataset with known format and known bbox locations, **When**
   infer/validate/remap simulation runs, **Then** the report confirms expected inferred format and
   bbox-check outcomes and no converted annotation files are written.

---

### User Story 2 - Infer Format and Run Conversion from Web GUI (Priority: P1)

A user uses a browser-accessible Python web GUI (defaulting to Streamlit unless an
alternative is justified in planning) that binds to localhost only, with no built-in user
authentication or role management, to scan a dataset directory, confirm inferred format, configure
conversion/remap options, and run the same behavior exposed by CLI commands. Concurrent runs to
the same output path are allowed, and the last completed run owns final output artifacts.

**Why this priority**: Ease of use for non-CLI workflows is required while keeping v1 scope to a
single local user and preserving strict local access controls for a no-authentication GUI.

**Independent Test**: Can be tested by launching the web GUI on localhost, completing one
conversion workflow, and verifying that the generated run configuration can be replayed via CLI
with equivalent output.

**Acceptance Scenarios**:

1. **Given** a dataset directory, **When** the user clicks infer format, **Then** the UI shows
   top candidate format, confidence, evidence, and ambiguity warnings.
2. **Given** an ambiguous inference result, **When** the user does not force a selection,
   **Then** conversion is blocked and the UI explains how to resolve the ambiguity.
3. **Given** a completed GUI run, **When** the user exports run details, **Then** the exported
   configuration can reproduce the same conversion output from CLI.
4. **Given** two GUI runs target the same output path concurrently, **When** both complete,
   **Then** final artifacts reflect the run that finished last and both reports explicitly note
   output-path contention and overwrite outcome.
5. **Given** the GUI service is running, **When** a client attempts non-localhost access,
   **Then** the connection is rejected because the GUI is bound to localhost only.

---

### User Story 3 - Import Datasets from External Sources (Priority: P2)

A user imports annotation datasets from external sources (Kaggle, Roboflow, GitHub, or a direct
URL), then validates and converts them using the same pipeline.

**Why this priority**: External source ingestion reduces manual setup and enables faster dataset
preparation workflows in the single-user v1 scope.

**Independent Test**: Can be tested by importing fixtures or controlled sample archives from each
supported source type and verifying provenance metadata, validation results, and conversion output.

**Acceptance Scenarios**:

1. **Given** a valid external source reference, **When** import is started, **Then** the dataset
   is downloaded to the configured workspace and provenance metadata is recorded.
2. **Given** a source that cannot be accessed or validated, **When** import runs, **Then** the
   run fails with actionable error details and no partial conversion is executed by default.
3. **Given** a successful import, **When** conversion is run from imported data, **Then** the
   workflow behaves identically to conversion from local datasets.
4. **Given** a direct URL import uses `http://` or `file://`, **When** import is started, **Then**
   the UI/CLI shows an explicit warning before retrieval continues and the import report records
   the protocol used.

---

### User Story 4 - Review Quality and Change Traceability (Priority: P3)

A reviewer verifies that each behavior change is covered by tests and linked to explicit specs so
conversion behavior remains understandable and safe to modify.

**Why this priority**: The tool handles high-volume data transformations where regressions can
silently affect model quality.

**Independent Test**: Can be tested by reviewing CI outputs and PR metadata for one feature change
to confirm required tests, quality gates, and traceability links are present.

**Acceptance Scenarios**:

1. **Given** a pull request that changes conversion logic, **When** CI runs, **Then** required
   tests and static checks must pass before merge.
2. **Given** a bug fix, **When** the change is reviewed, **Then** a regression test exists and
   references the corrected failure mode.

### Edge Cases

- Mixed-format directories containing both COCO and YOLO artifacts.
- Missing or unreadable image files needed for bbox bounds checks.
- YOLO rows with invalid token counts, non-numeric values, or out-of-range normalized values.
- COCO annotations with missing category IDs or malformed bbox arrays.
- Class maps that include destination IDs outside allowed taxonomy.
- Duplicate or conflicting imported files from multiple external sources.
- External archives with unsupported compression or directory nesting depth.
- Interrupted imports (network failures, timeouts) leaving partial datasets.
- Concurrent runs writing to the same output path where one run overwrites another.
- GUI bind address misconfigured to a non-localhost interface.
- Future multi-user editing conflicts are out of scope for v1 and deferred.
- `file://` imports that reference disallowed paths or fail path-safety checks.
- Dry-run execution accidentally writes converted annotation files.

### Assumptions

- v1 targets a single local GUI user; multi-user collaboration is deferred to later releases.
- GUI has no built-in authentication or role system in v1.
- When two runs target the same output path, parallel execution is allowed and last write wins.
- Direct URL imports may use `https://`, `http://`, or `file://`; `http://` and `file://` require
  warning messages before retrieval.
- External data access credentials are provided by the operating environment or secure app
  configuration and are not hard-coded.
- Imported external datasets are copied into a local workspace before conversion.
- v1 focuses on 2D bounding-box annotations and does not include segmentation or keypoints.
- Web GUI implementation follows the constitution: Python-based and browser-accessible, with
  Streamlit as the default unless an alternative is explicitly justified in the feature plan.
- User can provide sample datasets with known format and bbox expectations for dry-run verification.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support conversion between at least COCO and YOLO bounding-box formats.
- **FR-002**: System MUST expose all conversion, remapping, inference, validation, and reporting
  capabilities through CLI commands.
- **FR-003**: System MUST provide a browser-accessible Python web GUI that executes the same core
  operations and produces behavior equivalent to CLI for the same run configuration.
- **FR-003a**: System MUST use Streamlit as the default GUI framework unless an alternative is
  explicitly justified and approved in the implementation plan.
- **FR-003b**: System MUST NOT require built-in user authentication or role-based authorization in
  v1; GUI service MUST bind to localhost interfaces only by default.
- **FR-004**: System MUST infer likely source annotation format from directory scans and return
  confidence score, ranked candidates, and human-readable evidence.
- **FR-005**: System MUST block conversion on ambiguous inference unless user explicitly overrides.
- **FR-006**: System MUST support class remapping from source integer class IDs to destination
  integer class IDs.
- **FR-007**: System MUST support explicit drop semantics for one or more class IDs during remap.
- **FR-008**: System MUST support configurable unmapped-label policy with options `error`, `drop`,
  and `identity`.
- **FR-009**: System MUST generate a run report that includes class counts before/after remap,
  dropped labels, unmapped labels, invalid annotations, and output locations.
- **FR-010**: System MUST support dry-run mode that performs inference/validation/remap simulation
  without writing converted annotations.
- **FR-010a**: Dry-run mode MUST support verification against user-provided sample datasets with
  known format and bbox expectations, and MUST fail with actionable diagnostics if inferred format
  or bbox checks do not match expected results.
- **FR-010b**: System MUST support YAML dry-run expectation manifests for sample datasets,
  including dataset path, expected inferred source format, and expected bbox checks.
- **FR-011**: System MUST validate annotation schema and bbox constraints before writing output.
- **FR-012**: System MUST support importing datasets from Kaggle references.
- **FR-013**: System MUST support importing datasets from Roboflow references.
- **FR-014**: System MUST support importing datasets from GitHub repositories or release assets.
- **FR-015**: System MUST support importing datasets from user-provided direct URLs.
- **FR-015a**: System MUST allow `https://`, `http://`, and `file://` direct URL imports; imports
  using `http://` or `file://` MUST emit explicit warnings and record protocol in provenance.
- **FR-016**: System MUST capture provenance metadata for every imported dataset, including source
  reference, import timestamp, and integrity validation outcome.
- **FR-017**: System MUST prevent silent destructive changes; skipped, clipped, or dropped data
  MUST be explicitly counted and reported.
- **FR-018**: System MUST preserve deterministic output ordering for identical input/configuration.
- **FR-019**: System MUST persist GUI-created run configurations so the same user can rerun jobs.
- **FR-019a**: System MUST allow concurrent runs targeting the same output path; final output
  artifacts MUST reflect the run that finishes last, and both run reports MUST record contention
  and overwrite outcome.
- **FR-019b**: System MUST keep run configuration and report artifacts portable and schema-versioned
  so future multi-user workflows can be added without changing existing run artifact formats; during
  v1.x, readers MUST accept artifacts from the current v1 minor and the immediately previous v1
  minor version (N and N-1).
- **FR-020**: System MUST provide explicit error messages and non-zero exit codes for failed CLI
  operations.
- **FR-021**: System MUST provide audit-friendly run identifiers that link conversion outputs,
  reports, and source configurations.

### External Data Sources & Provenance *(include when feature ingests external data)*

- Source types allowed: Kaggle, Roboflow, GitHub, and user-provided URLs.
- Direct URL protocol and warning policy MUST follow **FR-015a**.
- Provenance fields required in reports: source reference, retrieval timestamp, artifact name,
  artifact checksum status, and import job identifier.
- Validation expectations: archive integrity checks, expected file-type checks, and schema checks
  before conversion starts.
- Failure behavior: fail closed by default; no conversion starts for failed imports unless user
  reruns after correction.

### Key Entities *(include if feature involves data)*

- **AnnotationDataset**: A logical dataset containing image records, annotation records, category
  definitions, and source metadata.
- **AnnotationRecord**: A single labeled bounding-box item with class ID, coordinates, and
  associated image reference.
- **ClassMap**: Mapping definition from source class IDs to destination class IDs or drop actions.
- **InferenceResult**: Ranked format candidates with confidence scores and evidence.
- **DryRunSampleManifest**: YAML manifest describing dry-run sample expectations, including dataset
  path, expected inferred format, bbox checks, and no-write assertions.
- **ConversionRun**: A reproducible execution request with source, destination, mapping, policies,
  status, and output artifact references.
- **ImportSource**: External source descriptor (provider type + location reference + auth context).
- **ImportArtifact**: Retrieved files and metadata from an import job, including validation status
  and provenance fields.
- **RunReport**: Human-readable and machine-readable summary of validation, remap, conversion, and
  import outcomes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In controlled fixture tests, 100% of valid annotations are converted without loss
  except for labels intentionally dropped by configuration.
- **SC-002**: For reference datasets with known formats, format inference top-1 accuracy is at
  least 95%, and ambiguous cases are reported instead of silently misclassified.
- **SC-003**: At least 95% of conversion runs on validated datasets complete without manual file
  edits after configuration is provided.
- **SC-004**: For the same input and run configuration, CLI and GUI outputs are equivalent in file
  content and report totals for 100% of regression fixtures.
- **SC-005**: External imports from each supported source type complete with recorded provenance in
  at least one automated integration fixture per source.
- **SC-006**: Core domain and adapter modules maintain at least 90% automated test coverage.
- **SC-007**: 100% of merged behavior-changing pull requests include linked spec references and at
  least one relevant test update.
- **SC-008**: In concurrent-run tests targeting one output path, 100% of run reports identify
  contention and which run produced the final artifacts.
- **SC-009**: In deployment tests, GUI listeners are bound to localhost only in 100% of default
  startup configurations.
- **SC-010**: In direct URL import tests using `http://` and `file://`, 100% of runs show
  protocol warnings and record protocol in provenance metadata.
- **SC-011**: In dry-run tests using user-provided sample datasets with known format and bbox
  locations, 100% of runs report expected inference/bbox-check outcomes and create zero converted
  annotation output files.
