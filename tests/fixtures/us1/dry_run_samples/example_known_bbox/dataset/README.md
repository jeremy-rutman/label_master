# Example Dataset Placeholder

`../manifest.yaml` currently points to `../../coco_minimal` (the Label Studio sample),
so this local `dataset/` directory is not used by default.

If you want this sample to be fully self-contained, update `dataset_root` in
`../manifest.yaml` back to `./dataset` and place:

- `annotations.json`
- `images/`
