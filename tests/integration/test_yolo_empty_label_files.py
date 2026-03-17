from __future__ import annotations

from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion


def test_yolo_export_writes_empty_label_file_for_unannotated_image(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "converted"

    result = execute_conversion(
        ConvertRequest(
            run_id="empty-label-check",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            class_map={4: None},
            unmapped_policy=UnmappedPolicy.IDENTITY,
        )
    )
    assert result.report.status == "completed"

    empty_label = output / "labels" / "90f04c9f-Drone_Detection_screenshot_05.12.2025.txt"
    populated_label = output / "labels" / "46c32d34-example_drone_picture_bw.txt"

    assert empty_label.exists()
    assert empty_label.read_text(encoding="utf-8") == ""
    assert populated_label.exists()
    assert populated_label.read_text(encoding="utf-8").strip()
