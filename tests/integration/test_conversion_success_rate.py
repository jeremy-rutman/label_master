from __future__ import annotations

from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.infra.config import load_mapping_file


def test_conversion_success_rate_at_least_95_percent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scenarios = [
        ConvertRequest(
            run_id="scn-1",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn1",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-2",
            input_path=Path("tests/fixtures/us1/yolo_minimal"),
            output_path=tmp_path / "scn2",
            src_format=SourceFormat.YOLO,
            dst_format=SourceFormat.COCO,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-3",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn3",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            class_map=load_mapping_file(Path("tests/fixtures/us1/maps/class_map_drop.yaml")),
            unmapped_policy=UnmappedPolicy.DROP,
            dry_run=False,
        ),
        ConvertRequest(
            run_id="scn-4",
            input_path=Path("tests/fixtures/us1/coco_minimal"),
            output_path=tmp_path / "scn4",
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=True,
        ),
    ]

    successes = 0
    for scenario in scenarios:
        try:
            result = execute_conversion(scenario)
            if result.report.status == "completed":
                successes += 1
        except Exception:
            pass

    rate = successes / len(scenarios)
    assert rate >= 0.95, f"conversion success rate below target: {rate:.3%}"
