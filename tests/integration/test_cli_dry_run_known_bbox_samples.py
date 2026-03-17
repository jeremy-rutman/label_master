from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
DRY_RUN_ROOT = Path("tests/fixtures/us1/dry_run_samples")


def test_known_bbox_example_manifest_verification_passes() -> None:
    manifest = DRY_RUN_ROOT / "example_known_bbox" / "manifest.yaml"
    result = RUNNER.invoke(app, ["verify-dry-run-manifest", "--manifest", str(manifest)])
    assert result.exit_code == 0


def test_known_bbox_laser_turret_manifests_when_available() -> None:
    manifests = sorted((DRY_RUN_ROOT / "laser_turret_provided").glob("*/manifest.yaml"))
    assert manifests

    ran_any = False
    for manifest in manifests:
        dataset_root = manifest.read_text(encoding="utf-8")
        if "dataset_root:" not in dataset_root:
            continue

        # Skip unavailable user-local datasets while keeping fixture contract testable.
        import yaml

        payload = yaml.safe_load(dataset_root)
        root = Path(payload["dataset_root"])
        if not root.exists():
            continue

        result = RUNNER.invoke(app, ["verify-dry-run-manifest", "--manifest", str(manifest)])
        assert result.exit_code == 0, f"manifest failed: {manifest} -> {result.stdout}"
        ran_any = True

    if not ran_any:
        pytest.skip("No laser_turret_provided dataset roots are available on this machine")
