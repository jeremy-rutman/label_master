# GUI Workflow Contract

**Feature**: `002-enhance-gui`  
**Interface Type**: Streamlit GUI behavior contract

## Workflow Surface

Tabs:
- `1. Dataset`
- `2. Format & Preview`
- `3. Output`
- `4. Label Mapping`
- `5. Review & Run`

## Dataset Tab Contract

### Controls

- `Input directory` text input
- `Browse...` input-directory action button

### Behavior

- Clicking `Browse...` MUST attempt directory selection.
- If browse succeeds, selected path MUST populate `Input directory`.
- If browse is unavailable/fails, UI MUST show explanatory message and preserve manual input path.
- Path validation MUST surface actionable messages (`missing`, `not directory`, `unreadable`).
- Path validation state MUST persist across reruns/tab changes in the same session.

## Label Mapping Tab Contract

### Row Shape

Each mapping row MUST support:
- `source_class_id` (text)
- `action` (`map` or `drop`)
- `destination_class_id` (text; required for `map`)

### Behavior

- Row parsing MUST be deterministic and produce row-level errors for invalid rows.
- Duplicate `source_class_id` rows MUST fail validation.
- Invalid mapping rows MUST block run execution in Review & Run.
- Valid mappings MUST be displayed as normalized class-map payload preview.

## Review & Run Contract

### Run-Blocking Conditions

Run action MUST be disabled if any of the following are true:
- Input directory is empty/invalid/unreadable.
- Output directory is empty.
- Source or destination format selection is invalid.
- Mapping parser reports one or more errors.

### Run Status and Progress

- Review & Run MUST expose explicit status values: `idle`, `running`, `completed`, `failed`.
- Progress presentation MUST map status to deterministic values (`0`, intermediate while running, `100` on terminal states).

### Run Success Artifacts

On successful run:
- GUI MUST persist run config JSON.
- If mapping rows are non-empty, GUI MUST persist a run-specific mapping artifact.
- GUI MUST display summary metrics from run report:
  - processed images
  - converted annotations
  - warning/error totals

### Output Access

- GUI MUST expose an output-directory access action.
- If auto-open fails, GUI MUST still show the absolute output path.

## CLI Parity Rule

GUI-triggered conversion inputs MUST map to equivalent CLI invocation parameters and produce
matching output content/report totals for the same dataset and mapping artifact.
