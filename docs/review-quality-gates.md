# Review Quality Gates

This project enforces review and governance checks for behavior-changing work.

## Required checks

- `ruff check .`
- `mypy src`
- `pytest --cov=src/label_master --cov-report=term-missing`
- `python scripts/ci/check_spec_traceability.py --evidence <evidence-file>`
- `python scripts/ci/check_regression_tests.py --evidence <evidence-file>`

## Evidence policy

- Every behavior change must reference a spec path under `specs/`.
- Every behavior change must reference at least one test path under `tests/`.
- If `Bugfix: true` is declared in evidence, a `Regression-Test: tests/...` line is mandatory.
