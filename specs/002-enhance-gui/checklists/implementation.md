# Implementation Checklist Outcomes

Date: 2026-03-05

## Commands Run

1. `PYTHONPATH=src .venv/bin/python -m pytest -q -o cache_dir=/tmp/pytest-cache`  
Result: PASS (`74 passed`, `1 skipped`)

2. `RUFF_CACHE_DIR=/tmp/ruff-cache .venv/bin/ruff check .`  
Result: PASS

3. `MYPY_CACHE_DIR=/tmp/mypy-cache PYTHONPATH=src .venv/bin/python -m mypy src`  
Result: PASS (`Success: no issues found in 42 source files`)

## Feature Test Evidence

- Added/updated tests covering browse success/cancel/unavailable fallback, input blocking for invalid and unreadable paths, mapping-row blocking and deterministic errors, run-status transitions (`idle/running/completed/failed`), summary metrics rendering, output-directory open fallback visibility, and GUI class-map schema contract validation.
- GUI mapping artifact schema validated with fixture payloads and real persisted `*.gui.class_map.json` output.

## SC-001 Timing Evidence

Measured on fixture `tests/fixtures/us1/coco_minimal` using helper-flow timing (browse/manual path selection -> validation -> preview readiness):

- Browse-assisted path selection flow: `0.0006s`
- Manual-entry fallback flow (browse unavailable): `0.0003s`

Both measured flows are far below the SC-001 threshold of 60 seconds and reached valid preview state without validation errors.

## Notes

- This environment has unwritable default cache directories (`.pytest_cache`, `.ruff_cache`, `.mypy_cache`), so quality-gate commands were run with cache directories redirected to `/tmp`.
- Hypothesis emitted non-blocking warnings about fallback in-memory example database due unwritable `.hypothesis/examples`.
