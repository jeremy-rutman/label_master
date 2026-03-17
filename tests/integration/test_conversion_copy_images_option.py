from __future__ import annotations

from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion


def _image_files(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])


def test_convert_copy_images_enabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_copy"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-enabled",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    output_images = _image_files(output)
    source_images = _image_files(source / "images")
    assert output_images
    assert [path.name for path in output_images] == [path.name for path in source_images]


def test_convert_copy_images_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_no_copy"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-disabled",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=False,
        )
    )

    assert result.report.status == "completed"
    output_images = _image_files(output)
    assert output_images == []
