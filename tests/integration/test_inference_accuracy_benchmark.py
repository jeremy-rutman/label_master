from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from label_master.core.services.infer_service import infer_format

MANIFEST_ROOT = Path("tests/fixtures/us1/dry_run_samples")


def test_inference_accuracy_benchmark_top1_at_least_95_percent() -> None:
    manifests = sorted(MANIFEST_ROOT.rglob("manifest.yaml"))
    assert manifests

    total = 0
    correct = 0
    for manifest in manifests:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        dataset_root = Path(payload["dataset_root"])
        if not dataset_root.is_absolute():
            dataset_root = (manifest.parent / dataset_root).resolve()
        if not dataset_root.exists():
            continue

        total += 1
        expected_top = payload["expected"]["inference"]["top_candidate"]
        inferred = infer_format(dataset_root, force=True)
        top_candidate = inferred.candidates[0].format.value
        if top_candidate == expected_top:
            correct += 1

    if total == 0:
        pytest.skip("No available datasets found for inference benchmark")

    accuracy = correct / total
    assert accuracy >= 0.95, f"top-1 accuracy below target: {accuracy:.3%}"
