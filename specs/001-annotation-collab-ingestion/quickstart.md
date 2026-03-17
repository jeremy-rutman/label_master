# Quickstart: Bounding-Box Conversion and Single-User Web GUI Ingestion

## Prerequisites

- Python 3.11+
- Local filesystem access to dataset input/output paths
- Optional credentials configured for Kaggle/Roboflow/GitHub imports

## 1) Environment Setup

```bash
cd /home/jeremy/data/drone_detect/label_master
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Infer and Validate a Dataset

```bash
annobox infer --input ./data/raw_dataset --json
annobox validate --input ./data/raw_dataset --format auto --strict
```

Expected:
- Inference emits candidate formats with confidence and evidence.
- Validation exits non-zero if strict checks fail.

## 3) Convert with Remap (Dry Run First)

```bash
annobox convert \
  --input ./data/raw_dataset \
  --output ./data/converted_dataset \
  --src auto \
  --dst yolo \
  --map ./maps/classes_v2.yaml \
  --unmapped-policy error \
  --dry-run
```

Then execute without `--dry-run` once the report is clean.

### Prepare known-bbox dry-run sample manifests

```bash
mkdir -p tests/fixtures/us1/dry_run_samples/my_sample
cp tests/fixtures/us1/dry_run_samples/manifest.template.yaml \
  tests/fixtures/us1/dry_run_samples/my_sample/manifest.yaml
```

Then place your dataset under `tests/fixtures/us1/dry_run_samples/my_sample/dataset/` and fill
known format + bbox expectations in `manifest.yaml`.

Ready-made local manifests for your provided Laser Turret datasets are in:
`tests/fixtures/us1/dry_run_samples/laser_turret_provided/`.

Validate a manifest directly:

```bash
annobox verify-dry-run-manifest \
  --manifest tests/fixtures/us1/dry_run_samples/example_known_bbox/manifest.yaml
```

## 4) Import External Data

### Kaggle example

```bash
annobox import --provider kaggle --source-ref owner/dataset-name --output ./data/imports/kaggle_ds
```

### Direct URL examples

```bash
annobox import --provider direct_url --source-ref https://example.org/dataset.zip --output ./data/imports/url_https
annobox import --provider direct_url --source-ref http://example.org/dataset.zip --output ./data/imports/url_http
annobox import --provider direct_url --source-ref file:///tmp/local-dataset.zip --output ./data/imports/url_file
```

Expected:
- `http://` and `file://` imports show warnings before retrieval.
- Import provenance records protocol and integrity status.

## 5) Launch GUI (localhost-only)

```bash
streamlit run src/label_master/interfaces/gui/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` locally.

Expected:
- GUI is not reachable from non-localhost interfaces.
- GUI actions map to same core operations/reports as CLI.

## 6) Verify Concurrent Output-Path Behavior

Run two conversions against the same output directory concurrently.

Expected:
- Last completed run owns final output files.
- Both run reports include contention metadata and overwrite outcome.

## 7) Quality Gates

```bash
ruff check .
mypy src
pytest -q --maxfail=1
pytest --cov=src/label_master --cov-report=term-missing
```

Expected:
- All checks pass.
- Core domain + adapters coverage remains >= 90%.
