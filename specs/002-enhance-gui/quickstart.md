# Quickstart: Streamlit GUI Workflow and Mapping Reliability Enhancements

## Prerequisites

- Python 3.11+
- Local filesystem access to fixture datasets
- Streamlit runtime on localhost

## 1) Environment Setup

```bash
cd /home/jeremy/Documents/label_master
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Launch GUI

```bash
streamlit run src/label_master/interfaces/gui/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` locally.

## 3) Use Input Directory Browse Flow

1. Open `1. Dataset` tab.
2. Click `Browse...` and select `tests/fixtures/us1/coco_minimal`.
3. Confirm the input field is populated and validation shows success.
4. If browse is unavailable in your environment, enter the path manually and verify fallback message.

## 4) Configure Mapping and Run Conversion

1. In `4. Label Mapping`, add rows:
   - `source_class_id=2`, `action=drop`
   - `source_class_id=3`, `action=map`, `destination_class_id=10`
2. Confirm there are no row-level validation errors and the normalized class-map preview appears.
3. In `5. Review & Run`, verify run status starts at `idle` and blocking errors are absent.
4. Run conversion and confirm status transitions to `running` then `completed`.
5. Confirm summary metrics cards render for processed images, converted labels, warnings, and errors.
6. Click `Open output directory` and verify either auto-open succeeds or fallback path messaging is shown.

## 5) Validate Mapping Artifacts and Parity

Check that a run-specific map artifact exists in the selected output directory:

```bash
find /tmp/label_master_gui_output -maxdepth 1 -type f -name "*.gui.class_map.json"
```

Validate run configuration/report artifacts are present in selected output directory:

```bash
find /tmp/label_master_gui_output -maxdepth 1 -type f -name "*.gui.config.json"
find /tmp/label_master_gui_output -maxdepth 1 -type f -name "*.report.json"
```

## 6) Run Quality Gates

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -o cache_dir=/tmp/pytest-cache
RUFF_CACHE_DIR=/tmp/ruff-cache .venv/bin/ruff check .
MYPY_CACHE_DIR=/tmp/mypy-cache PYTHONPATH=src .venv/bin/python -m mypy src
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_gui_viewmodels.py -q -o cache_dir=/tmp/pytest-cache
PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_gui_inline_mapping_persistence.py -q -o cache_dir=/tmp/pytest-cache
```

Expected:
- Mapping validation errors block runs.
- Valid mapping rows are persisted and affect output annotations.
- GUI/CLI parity fixtures remain green.
- Run summary metrics and output-directory action fallback are visible post-run.
