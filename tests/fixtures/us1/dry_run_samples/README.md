# Dry-Run Sample Dataset Manifests

This directory stores user-provided dry-run samples with known source format and known bounding
box expectations.

## Expected layout

```text
tests/fixtures/us1/dry_run_samples/
├── manifest.template.yaml
└── <sample_id>/
    ├── manifest.yaml
    └── dataset/
        └── <dataset files>
```

## How to add a sample

1. Copy `manifest.template.yaml` to `<sample_id>/manifest.yaml`.
2. Set `dataset_root`:
   - Preferred: `./dataset` with files under `<sample_id>/dataset/`
   - Also allowed: absolute dataset path (for user-provided external dataset roots)
3. Fill expected format and bbox checks with known values.

## Manifest field requirements

- `manifest_version`: integer format version for the manifest itself.
- `sample_id`: unique sample name.
- `dataset_root`: relative or absolute path to the dataset directory.
- `expected.source_format`: one of `coco` or `yolo`.
- `expected.inference.top_candidate`: expected top inferred format.
- `bbox_checks.expected_boxes`: known bounding boxes to validate in dry-run.
- `dry_run_expectations.expect_converted_outputs_written`: must stay `false`.

## Notes

- Prefer relative paths for portability; absolute paths are acceptable for local-only validation.
- Prefer small samples that still cover each class and edge behavior.
- Include at least one bbox per class to catch remap/inference regressions.

## Included ready-to-use manifests

- `laser_turret_provided/` contains manifests wired to:
  `/home/jeremy/data/drone_detect/laser_turret/yolo_models/datasets/`
